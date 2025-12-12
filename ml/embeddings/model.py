import logging
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model optimized for biomedical text
DEFAULT_MODEL = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"


class EmbeddingModel:
    """PubMedBERT-based embedding model"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: SentenceTransformer | None = None
        self._dimension: int = 768

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load model"""
        if self._model is None:
            logger.info(f"Loading model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self._dimension}")
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        if self._model is None:
            # Load model to get dimension
            _ = self.model
        return self._dimension

    def encode(
        self,
        text: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode single text to embedding

        Args:
            text: Input text
            normalize: Whether to L2 normalize the embedding

        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding  # type: ignore

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode batch of texts to embeddings

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings
            show_progress: Show progress bar

        Returns:
            Embedding matrix as numpy array (n_texts x dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings  # type: ignore

    def to_list(self, embedding: np.ndarray) -> list[float]:
        """Convert numpy embedding to list"""
        return embedding.tolist()


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Get singleton embedding model instance"""
    return EmbeddingModel()
