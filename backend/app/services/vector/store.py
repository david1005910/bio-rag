import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Vector search result"""

    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any]


class VectorStore:
    """ChromaDB vector store for paper chunks"""

    COLLECTION_NAME = "paper_chunks"

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        persist_directory: str | None = None,
    ) -> None:
        self.host = host or settings.CHROMA_HOST
        self.port = port or settings.CHROMA_PORT
        self.persist_directory = persist_directory

        self._client: chromadb.Client | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def client(self) -> chromadb.Client:
        """Get ChromaDB client (lazy initialization)"""
        if self._client is None:
            if self.persist_directory:
                # Local persistent client
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )
            else:
                # HTTP client for remote ChromaDB
                self._client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    settings=Settings(anonymized_telemetry=False),
                )
            logger.info(f"ChromaDB client initialized: {self.host}:{self.port}")
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create collection"""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:M": 16,
                    "hnsw:ef_construction": 200,
                },
            )
            logger.info(f"Collection '{self.COLLECTION_NAME}' ready")
        return self._collection

    def add(
        self,
        chunk_id: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add single document to vector store

        Args:
            chunk_id: Unique chunk identifier
            content: Text content
            embedding: Vector embedding
            metadata: Additional metadata
        """
        self.collection.add(
            ids=[chunk_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata or {}],
        )

    def add_batch(
        self,
        chunk_ids: list[str],
        contents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Add batch of documents to vector store

        Args:
            chunk_ids: List of unique chunk identifiers
            contents: List of text contents
            embeddings: List of vector embeddings
            metadatas: List of metadata dicts
        """
        if not chunk_ids:
            return

        self.collection.add(
            ids=chunk_ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(chunk_ids),
        )
        logger.info(f"Added {len(chunk_ids)} documents to vector store")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search for similar documents

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Metadata filter (ChromaDB where clause)
            min_score: Minimum similarity score threshold

        Returns:
            List of SearchResult objects
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []

        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            for i, chunk_id in enumerate(ids):
                # Convert distance to similarity score (cosine distance to similarity)
                score = 1 - distances[i] if distances else 0

                if score >= min_score:
                    search_results.append(
                        SearchResult(
                            chunk_id=chunk_id,
                            content=documents[i] if i < len(documents) else "",
                            score=score,
                            metadata=metadatas[i] if i < len(metadatas) else {},
                        )
                    )

        return search_results

    def search_by_text(
        self,
        query_text: str,
        embedding_func: Callable,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search using text query (will be embedded)

        Args:
            query_text: Query text
            embedding_func: Function to generate embedding from text
            top_k: Number of results
            filter_metadata: Metadata filter

        Returns:
            List of SearchResult objects
        """
        query_embedding = embedding_func(query_text)
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        """Get document by ID"""
        results = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas", "embeddings"],
        )

        if results["ids"]:
            return {
                "id": results["ids"][0],
                "document": results["documents"][0] if results["documents"] else None,
                "metadata": results["metadatas"][0] if results["metadatas"] else None,
                "embedding": results["embeddings"][0] if results["embeddings"] else None,
            }
        return None

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete documents by IDs"""
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} documents")

    def delete_by_metadata(self, filter_metadata: dict[str, Any]) -> None:
        """Delete documents matching metadata filter"""
        self.collection.delete(where=filter_metadata)

    def count(self) -> int:
        """Get total document count"""
        return self.collection.count()

    def get_all_documents(self, limit: int = 10000) -> list[dict[str, Any]]:
        """
        Get all documents from collection

        Args:
            limit: Maximum number of documents to return

        Returns:
            List of document dicts with chunk_id, content, and metadata
        """
        results = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"],
        )

        documents: list[dict[str, Any]] = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                documents.append({
                    "chunk_id": chunk_id,
                    "content": results["documents"][i] if results["documents"] else "",
                    "pmid": results["metadatas"][i].get("pmid", "") if results["metadatas"] else "",
                    "title": results["metadatas"][i].get("title", "") if results["metadatas"] else "",
                    "section": results["metadatas"][i].get("section") if results["metadatas"] else None,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        return documents

    def clear(self) -> None:
        """Clear all documents from collection"""
        # Delete and recreate collection
        self.client.delete_collection(self.COLLECTION_NAME)
        self._collection = None
        logger.info("Vector store cleared")


# Singleton instance - use persistent directory if host not configured
_persist_dir = settings.CHROMA_PERSIST_DIR if not settings.CHROMA_HOST else None
vector_store = VectorStore(persist_directory=_persist_dir)
