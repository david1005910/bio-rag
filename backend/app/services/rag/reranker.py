"""Cross-Encoder Re-ranker for improving retrieval relevance"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from app.services.rag.retriever import RetrievedDocument

logger = logging.getLogger(__name__)

# Default Cross-Encoder model for biomedical domain
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BIOMEDICAL_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


@dataclass
class RerankerConfig:
    """Configuration for re-ranker"""

    model_name: str = DEFAULT_RERANKER_MODEL
    top_k: int = 5
    batch_size: int = 32
    max_length: int = 512
    device: str | None = None
    score_threshold: float = 0.0


class CrossEncoderReranker:
    """Cross-Encoder based re-ranker for improving retrieval relevance"""

    def __init__(self, config: RerankerConfig | None = None) -> None:
        self.config = config or RerankerConfig()
        self._model: CrossEncoder | None = None
        self._device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def model(self) -> CrossEncoder:
        """Lazy load cross-encoder model"""
        if self._model is None:
            logger.info(f"Loading Cross-Encoder model: {self.config.model_name}")
            self._model = CrossEncoder(
                self.config.model_name,
                max_length=self.config.max_length,
                device=self._device,
            )
            logger.info(f"Cross-Encoder model loaded on device: {self._device}")
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Re-rank documents using cross-encoder

        Args:
            query: User query
            documents: List of documents to re-rank
            top_k: Number of top documents to return

        Returns:
            Re-ranked list of documents
        """
        if not documents:
            return []

        top_k = top_k or self.config.top_k
        top_k = min(top_k, len(documents))

        # Prepare query-document pairs
        pairs = [[query, doc.content] for doc in documents]

        # Score pairs with cross-encoder
        logger.debug(f"Re-ranking {len(pairs)} documents")
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Filter by threshold and take top_k
        results: list[RetrievedDocument] = []
        for doc, score in scored_docs[:top_k]:
            if score < self.config.score_threshold:
                continue

            # Create new document with updated score
            results.append(
                RetrievedDocument(
                    chunk_id=doc.chunk_id,
                    pmid=doc.pmid,
                    title=doc.title,
                    content=doc.content,
                    section=doc.section,
                    score=float(score),  # Use cross-encoder score
                    metadata={
                        **doc.metadata,
                        "original_score": doc.score,
                        "rerank_score": float(score),
                    },
                )
            )

        logger.debug(f"Re-ranking complete: {len(results)} documents after filtering")
        return results

    def rerank_with_scores(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[tuple[RetrievedDocument, float, float]]:
        """
        Re-rank and return original and re-ranked scores

        Args:
            query: User query
            documents: Documents to re-rank

        Returns:
            List of (document, original_score, rerank_score) tuples
        """
        if not documents:
            return []

        pairs = [[query, doc.content] for doc in documents]
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        results = [
            (doc, doc.score, float(score))
            for doc, score in zip(documents, scores)
        ]

        # Sort by rerank score
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def compute_relevance_score(
        self,
        query: str,
        document: str,
    ) -> float:
        """
        Compute relevance score for a single query-document pair

        Args:
            query: User query
            document: Document text

        Returns:
            Relevance score
        """
        score = self.model.predict([[query, document]])
        return float(score[0])

    def batch_score(
        self,
        queries: list[str],
        documents: list[str],
    ) -> np.ndarray:
        """
        Score multiple query-document pairs

        Args:
            queries: List of queries
            documents: List of documents (same length as queries)

        Returns:
            Array of scores
        """
        if len(queries) != len(documents):
            raise ValueError("Queries and documents must have same length")

        pairs = list(zip(queries, documents))
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )
        return np.array(scores)


class TwoStageRetriever:
    """Two-stage retrieval: fast initial retrieval + cross-encoder re-ranking"""

    def __init__(
        self,
        retriever_func: Callable,
        reranker: CrossEncoderReranker | None = None,
        initial_k: int = 50,
        final_k: int = 10,
    ) -> None:
        """
        Args:
            retriever_func: Function that takes (query, top_k) and returns documents
            reranker: Cross-encoder re-ranker
            initial_k: Number of candidates for initial retrieval
            final_k: Final number of documents after re-ranking
        """
        self._retriever_func = retriever_func
        self._reranker = reranker
        self.initial_k = initial_k
        self.final_k = final_k

    @property
    def reranker(self) -> CrossEncoderReranker:
        """Get re-ranker instance"""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """
        Two-stage retrieval

        Args:
            query: Search query
            top_k: Final number of documents
            filter_metadata: Metadata filter for initial retrieval

        Returns:
            Re-ranked documents
        """
        final_k = top_k or self.final_k

        # Stage 1: Fast initial retrieval
        initial_docs = self._retriever_func(
            query=query,
            top_k=self.initial_k,
            filter_metadata=filter_metadata,
        )
        logger.debug(f"Stage 1: Retrieved {len(initial_docs)} candidates")

        if not initial_docs:
            return []

        # Stage 2: Cross-encoder re-ranking
        reranked_docs = self.reranker.rerank(
            query=query,
            documents=initial_docs,
            top_k=final_k,
        )
        logger.debug(f"Stage 2: Re-ranked to {len(reranked_docs)} documents")

        return reranked_docs


# Singleton instances
cross_encoder_reranker = CrossEncoderReranker()


def create_two_stage_retriever(
    retriever_func: callable,
    initial_k: int = 50,
    final_k: int = 10,
    reranker_model: str | None = None,
) -> TwoStageRetriever:
    """
    Factory function to create two-stage retriever

    Args:
        retriever_func: Initial retrieval function
        initial_k: Candidates for initial retrieval
        final_k: Final documents after re-ranking
        reranker_model: Optional custom re-ranker model

    Returns:
        Configured TwoStageRetriever
    """
    config = RerankerConfig(
        model_name=reranker_model or DEFAULT_RERANKER_MODEL,
        top_k=final_k,
    )
    reranker = CrossEncoderReranker(config)
    return TwoStageRetriever(
        retriever_func=retriever_func,
        reranker=reranker,
        initial_k=initial_k,
        final_k=final_k,
    )
