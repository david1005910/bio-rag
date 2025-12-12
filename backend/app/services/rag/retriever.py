"""RAG retriever component for document retrieval"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.services.vector.store import SearchResult, VectorStore, vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """Retrieved document with metadata"""

    chunk_id: str
    pmid: str
    title: str
    content: str
    section: str | None
    score: float
    metadata: dict[str, Any]


class RAGRetriever:
    """Retriever for RAG pipeline"""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_func: Callable | None = None,
        top_k: int = 10,
        min_score: float = 0.5,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_func = embedding_func
        self.top_k = top_k
        self.min_score = min_score

    @property
    def store(self) -> VectorStore:
        """Get vector store"""
        if self._vector_store is None:
            self._vector_store = vector_store
        return self._vector_store

    def set_embedding_func(self, func: Callable) -> None:
        """Set embedding function"""
        self._embedding_func = func

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve relevant documents for a query

        Args:
            query: User query
            top_k: Number of documents to retrieve
            filter_metadata: Metadata filter

        Returns:
            List of RetrievedDocument objects
        """
        if self._embedding_func is None:
            raise ValueError("Embedding function not set")

        top_k = top_k or self.top_k

        # Generate query embedding
        query_embedding = self._embedding_func(query)

        # Search vector store
        results = self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
            min_score=self.min_score,
        )

        return self._convert_results(results)

    def retrieve_with_embedding(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve using pre-computed embedding

        Args:
            query_embedding: Pre-computed query embedding
            top_k: Number of documents to retrieve
            filter_metadata: Metadata filter

        Returns:
            List of RetrievedDocument objects
        """
        top_k = top_k or self.top_k

        results = self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
            min_score=self.min_score,
        )

        return self._convert_results(results)

    def _convert_results(
        self,
        results: list[SearchResult],
    ) -> list[RetrievedDocument]:
        """Convert SearchResult to RetrievedDocument"""
        documents: list[RetrievedDocument] = []

        for result in results:
            metadata = result.metadata or {}
            documents.append(
                RetrievedDocument(
                    chunk_id=result.chunk_id,
                    pmid=metadata.get("pmid", "unknown"),
                    title=metadata.get("title", "Unknown Title"),
                    content=result.content,
                    section=metadata.get("section"),
                    score=result.score,
                    metadata=metadata,
                )
            )

        return documents

    def get_unique_papers(
        self,
        documents: list[RetrievedDocument],
    ) -> list[str]:
        """Get unique PMIDs from retrieved documents"""
        seen: set[str] = set()
        unique: list[str] = []

        for doc in documents:
            if doc.pmid not in seen:
                seen.add(doc.pmid)
                unique.append(doc.pmid)

        return unique


# Singleton instance
rag_retriever = RAGRetriever()
