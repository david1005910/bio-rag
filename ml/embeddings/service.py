import logging
from dataclasses import dataclass

import numpy as np

from ml.embeddings.model import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""

    text: str
    embedding: list[float]
    token_count: int


class EmbeddingService:
    """Service for generating text embeddings"""

    def __init__(self, model: EmbeddingModel | None = None) -> None:
        self._model = model

    @property
    def model(self) -> EmbeddingModel:
        """Get embedding model (lazy loading)"""
        if self._model is None:
            self._model = get_embedding_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.dimension

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for single text

        Args:
            text: Input text

        Returns:
            Embedding as list of floats
        """
        text = self._preprocess(text)
        embedding = self.model.encode(text)
        return self.model.to_list(embedding)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Generate embeddings for batch of texts

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            List of embeddings
        """
        processed_texts = [self._preprocess(t) for t in texts]
        embeddings = self.model.encode_batch(
            processed_texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        return [self.model.to_list(emb) for emb in embeddings]

    def embed_with_metadata(self, text: str) -> EmbeddingResult:
        """
        Generate embedding with metadata

        Args:
            text: Input text

        Returns:
            EmbeddingResult with text, embedding, and token count
        """
        processed_text = self._preprocess(text)
        embedding = self.embed_text(processed_text)
        token_count = self._estimate_tokens(processed_text)

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            token_count=token_count,
        )

    def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score (0-1 for normalized embeddings)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        return float(np.dot(vec1, vec2))

    def find_most_similar(
        self,
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Find most similar embeddings to query

        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples, sorted by score descending
        """
        query_vec = np.array(query_embedding)
        candidate_matrix = np.array(candidate_embeddings)

        # Compute similarities
        similarities = np.dot(candidate_matrix, query_vec)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def _preprocess(self, text: str) -> str:
        """Preprocess text before embedding"""
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Truncate if too long (model max is typically 512 tokens)
        max_chars = 4000  # Approximate 512 tokens
        if len(text) > max_chars:
            text = text[:max_chars]

        return text

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4


# Singleton service instance
embedding_service = EmbeddingService()
