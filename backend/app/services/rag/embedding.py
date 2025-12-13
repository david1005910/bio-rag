"""Embedding service for RAG"""

import logging
from functools import lru_cache

from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """OpenAI Embedding service"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
    ) -> None:
        self.model = model
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """Get OpenAI client (lazy initialization)"""
        if self._client is None:
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info(f"Embedding client initialized: {self.model}")
        return self._client

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not text.strip():
            return []

        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter empty texts
        filtered_texts = [t for t in texts if t.strip()]
        if not filtered_texts:
            return []

        try:
            response = self.client.embeddings.create(
                input=filtered_texts,
                model=self.model,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise


# Singleton instance
embedding_service = EmbeddingService()


@lru_cache(maxsize=1000)
def get_embedding(text: str) -> tuple[float, ...]:
    """Cached embedding function (returns tuple for hashability)"""
    embedding = embedding_service.embed(text)
    return tuple(embedding)


def embed_text(text: str) -> list[float]:
    """Embedding function to be used by retriever"""
    return list(get_embedding(text))
