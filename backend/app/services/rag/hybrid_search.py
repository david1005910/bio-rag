"""Hybrid Search combining Dense and BM25 retrieval with RRF fusion"""

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from app.services.rag.retriever import RAGRetriever, RetrievedDocument, rag_retriever
from app.services.vector.store import VectorStore, vector_store

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search"""

    dense_weight: float = 0.7
    bm25_weight: float = 0.3
    rrf_k: int = 60  # RRF constant
    top_k: int = 10
    min_score: float = 0.3


class BM25Index:
    """BM25 index for sparse retrieval"""

    def __init__(self) -> None:
        self._documents: list[dict[str, Any]] = []
        self._index: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        """
        Build BM25 index from documents

        Args:
            documents: List of documents with 'content' and 'chunk_id' keys
        """
        self._documents = documents
        self._tokenized_corpus = [
            self._tokenize(doc.get("content", "")) for doc in documents
        ]
        self._index = BM25Okapi(self._tokenized_corpus)
        logger.info(f"Built BM25 index with {len(documents)} documents")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Search using BM25

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (document_index, score) tuples
        """
        if self._index is None:
            raise ValueError("BM25 index not built. Call build_index first.")

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Get top-k indices and scores
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

        return results

    def get_document(self, index: int) -> dict[str, Any]:
        """Get document by index"""
        return self._documents[index]

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on whitespace/punctuation
        text = text.lower()
        # Basic tokenization - split on non-alphanumeric
        tokens = []
        current_token = []
        for char in text:
            if char.isalnum():
                current_token.append(char)
            else:
                if current_token:
                    tokens.append("".join(current_token))
                    current_token = []
        if current_token:
            tokens.append("".join(current_token))
        return tokens

    @property
    def is_built(self) -> bool:
        """Check if index is built"""
        return self._index is not None

    @property
    def document_count(self) -> int:
        """Get number of documents in index"""
        return len(self._documents)


class HybridSearcher:
    """Hybrid search combining dense and sparse retrieval"""

    def __init__(
        self,
        dense_retriever: RAGRetriever | None = None,
        vector_store: VectorStore | None = None,
        config: HybridSearchConfig | None = None,
    ) -> None:
        self._dense_retriever = dense_retriever
        self._vector_store = vector_store
        self._bm25_index = BM25Index()
        self.config = config or HybridSearchConfig()

    @property
    def dense_retriever(self) -> RAGRetriever:
        """Get dense retriever"""
        if self._dense_retriever is None:
            self._dense_retriever = rag_retriever
        return self._dense_retriever

    @property
    def store(self) -> VectorStore:
        """Get vector store"""
        if self._vector_store is None:
            self._vector_store = vector_store
        return self._vector_store

    def set_embedding_func(self, func: Callable) -> None:
        """Set embedding function"""
        self.dense_retriever.set_embedding_func(func)

    def build_bm25_index(self, documents: list[dict[str, Any]] | None = None) -> None:
        """
        Build BM25 index from documents

        Args:
            documents: Optional list of documents. If None, loads from vector store.
        """
        if documents is None:
            # Load all documents from vector store
            documents = self.store.get_all_documents()

        self._bm25_index.build_index(documents)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """
        Perform hybrid search

        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Metadata filter (applied to dense search)

        Returns:
            List of RetrievedDocument objects with fused scores
        """
        top_k = top_k or self.config.top_k

        # Retrieve more candidates for fusion
        candidate_k = min(top_k * 3, 100)

        # Dense retrieval
        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=candidate_k,
            filter_metadata=filter_metadata,
        )
        logger.debug(f"Dense retrieval: {len(dense_results)} results")

        # BM25 retrieval (if index is built)
        bm25_results: list[tuple[int, float]] = []
        if self._bm25_index.is_built:
            bm25_results = self._bm25_index.search(query, top_k=candidate_k)
            logger.debug(f"BM25 retrieval: {len(bm25_results)} results")

        # Fuse results using RRF
        fused_results = self._rrf_fusion(dense_results, bm25_results, top_k)

        return fused_results

    def _rrf_fusion(
        self,
        dense_results: list[RetrievedDocument],
        bm25_results: list[tuple[int, float]],
        top_k: int,
    ) -> list[RetrievedDocument]:
        """
        Fuse results using Reciprocal Rank Fusion (RRF)

        Args:
            dense_results: Results from dense retrieval
            bm25_results: Results from BM25 (index, score)
            top_k: Number of final results

        Returns:
            Fused and sorted results
        """
        k = self.config.rrf_k
        chunk_scores: dict[str, float] = defaultdict(float)
        chunk_docs: dict[str, RetrievedDocument] = {}

        # Add dense scores
        for rank, doc in enumerate(dense_results, 1):
            rrf_score = self.config.dense_weight * (1 / (k + rank))
            chunk_scores[doc.chunk_id] += rrf_score
            chunk_docs[doc.chunk_id] = doc

        # Add BM25 scores
        for rank, (idx, _score) in enumerate(bm25_results, 1):
            bm25_doc = self._bm25_index.get_document(idx)
            chunk_id = bm25_doc.get("chunk_id", "")

            rrf_score = self.config.bm25_weight * (1 / (k + rank))
            chunk_scores[chunk_id] += rrf_score

            # If not in dense results, create RetrievedDocument
            if chunk_id not in chunk_docs:
                chunk_docs[chunk_id] = RetrievedDocument(
                    chunk_id=chunk_id,
                    pmid=bm25_doc.get("pmid", "unknown"),
                    title=bm25_doc.get("title", "Unknown"),
                    content=bm25_doc.get("content", ""),
                    section=bm25_doc.get("section"),
                    score=0.0,  # Will be updated
                    metadata=bm25_doc.get("metadata", {}),
                )

        # Sort by fused score and take top_k
        sorted_chunks = sorted(
            chunk_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        # Build final results with updated scores
        results: list[RetrievedDocument] = []
        for chunk_id, fused_score in sorted_chunks:
            doc = chunk_docs[chunk_id]
            # Update score to fused score (normalized)
            results.append(
                RetrievedDocument(
                    chunk_id=doc.chunk_id,
                    pmid=doc.pmid,
                    title=doc.title,
                    content=doc.content,
                    section=doc.section,
                    score=fused_score,
                    metadata=doc.metadata,
                )
            )

        return results

    def search_with_scores(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[RetrievedDocument, dict[str, float]]]:
        """
        Search with detailed score breakdown

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (document, scores_dict) with dense_score, bm25_score, fused_score
        """
        top_k = top_k or self.config.top_k
        candidate_k = min(top_k * 3, 100)

        # Get dense results
        dense_results = self.dense_retriever.retrieve(query=query, top_k=candidate_k)
        dense_scores: dict[str, float] = {
            doc.chunk_id: doc.score for doc in dense_results
        }

        # Get BM25 results
        bm25_scores: dict[str, float] = {}
        if self._bm25_index.is_built:
            bm25_results = self._bm25_index.search(query, top_k=candidate_k)
            # Normalize BM25 scores
            if bm25_results:
                max_bm25 = max(score for _, score in bm25_results)
                for idx, score in bm25_results:
                    doc = self._bm25_index.get_document(idx)
                    chunk_id = doc.get("chunk_id", "")
                    bm25_scores[chunk_id] = score / max_bm25 if max_bm25 > 0 else 0

        # Compute fused scores
        all_chunk_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
        results: list[tuple[RetrievedDocument, dict[str, float]]] = []

        for chunk_id in all_chunk_ids:
            d_score = dense_scores.get(chunk_id, 0.0)
            b_score = bm25_scores.get(chunk_id, 0.0)
            fused = (
                self.config.dense_weight * d_score
                + self.config.bm25_weight * b_score
            )

            # Get document
            doc = None
            for d in dense_results:
                if d.chunk_id == chunk_id:
                    doc = d
                    break

            if doc is None and self._bm25_index.is_built:
                # Find in BM25 index
                for i in range(self._bm25_index.document_count):
                    bm25_doc = self._bm25_index.get_document(i)
                    if bm25_doc.get("chunk_id") == chunk_id:
                        doc = RetrievedDocument(
                            chunk_id=chunk_id,
                            pmid=bm25_doc.get("pmid", "unknown"),
                            title=bm25_doc.get("title", "Unknown"),
                            content=bm25_doc.get("content", ""),
                            section=bm25_doc.get("section"),
                            score=fused,
                            metadata=bm25_doc.get("metadata", {}),
                        )
                        break

            if doc:
                results.append((
                    doc,
                    {
                        "dense_score": d_score,
                        "bm25_score": b_score,
                        "fused_score": fused,
                    },
                ))

        # Sort by fused score
        results.sort(key=lambda x: x[1]["fused_score"], reverse=True)
        return results[:top_k]


# Singleton instance
hybrid_searcher = HybridSearcher()
