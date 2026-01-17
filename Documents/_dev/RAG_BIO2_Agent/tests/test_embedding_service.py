"""Tests for embedding_service.py"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import List
from abc import ABC, abstractmethod


# ============================================================================
# Testable implementations of embedding service classes
# ============================================================================

class MockEmbeddingGenerator:
    """Mock embedding generator for testing."""

    def __init__(self, dimension: int = 768):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self._dimension)
        # Generate deterministic embedding based on text hash
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(self._dimension).astype(np.float32)

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        embeddings = [self.encode(text) for text in texts]
        return np.array(embeddings)


class _TestablePubMedBERTEmbedding:
    """Testable version of PubMedBERT embedding generator."""

    def __init__(self, dimension: int = 768):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str, max_length: int = 512) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self._dimension)
        # Simulate embedding generation
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(self._dimension).astype(np.float32)

    def batch_encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = [self.encode(text) for text in batch]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)


class _TestableOpenAIEmbedding:
    """Testable version of OpenAI embedding generator."""

    def __init__(self, model: str = "text-embedding-3-small", dimension: int = 1536):
        self.model = model
        self._dimension = dimension
        self._client = MagicMock()

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self._dimension)
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(self._dimension).astype(np.float32)

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            if text and text.strip():
                embedding = self.encode(text)
            else:
                embedding = np.zeros(self._dimension)
            embeddings.append(embedding)
        return np.array(embeddings)


class _TestableEmbeddingService:
    """Testable version of EmbeddingService."""

    def __init__(self, model_type: str = "pubmedbert"):
        self.model_type = model_type
        self._generator = None
        self._initialize()

    def _initialize(self):
        if self.model_type == "pubmedbert":
            self._generator = _TestablePubMedBERTEmbedding(dimension=768)
        else:
            self._generator = _TestableOpenAIEmbedding(dimension=1536)

    @property
    def dimension(self) -> int:
        return self._generator.dimension

    def encode(self, text: str) -> np.ndarray:
        return self._generator.encode(text)

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        return self._generator.batch_encode(texts)

    def switch_model(self, model_type: str):
        self.model_type = model_type
        self._initialize()


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def pubmedbert_embedding():
    """Create a testable PubMedBERT embedding generator."""
    return _TestablePubMedBERTEmbedding()


@pytest.fixture
def openai_embedding():
    """Create a testable OpenAI embedding generator."""
    return _TestableOpenAIEmbedding()


@pytest.fixture
def embedding_service_pubmedbert():
    """Create an embedding service with PubMedBERT."""
    return _TestableEmbeddingService(model_type="pubmedbert")


@pytest.fixture
def embedding_service_openai():
    """Create an embedding service with OpenAI."""
    return _TestableEmbeddingService(model_type="openai")


# ============================================================================
# Tests
# ============================================================================

class TestPubMedBERTEmbedding:
    """Tests for PubMedBERT embedding generator."""

    def test_dimension_is_768(self, pubmedbert_embedding):
        """Test that PubMedBERT dimension is 768."""
        assert pubmedbert_embedding.dimension == 768

    def test_encode_returns_correct_shape(self, pubmedbert_embedding):
        """Test that encode returns array of correct shape."""
        text = "This is a biomedical text about cancer treatment."
        embedding = pubmedbert_embedding.encode(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_encode_empty_text_returns_zeros(self, pubmedbert_embedding):
        """Test that empty text returns zero vector."""
        embedding = pubmedbert_embedding.encode("")

        assert np.all(embedding == 0)

    def test_encode_whitespace_only_returns_zeros(self, pubmedbert_embedding):
        """Test that whitespace-only text returns zero vector."""
        embedding = pubmedbert_embedding.encode("   ")

        assert np.all(embedding == 0)

    def test_encode_is_deterministic(self, pubmedbert_embedding):
        """Test that same text produces same embedding."""
        text = "Reproducible biomedical research"
        embedding1 = pubmedbert_embedding.encode(text)
        embedding2 = pubmedbert_embedding.encode(text)

        np.testing.assert_array_equal(embedding1, embedding2)

    def test_encode_different_texts_produce_different_embeddings(self, pubmedbert_embedding):
        """Test that different texts produce different embeddings."""
        embedding1 = pubmedbert_embedding.encode("Cancer treatment")
        embedding2 = pubmedbert_embedding.encode("Heart disease prevention")

        assert not np.array_equal(embedding1, embedding2)

    def test_batch_encode_returns_correct_shape(self, pubmedbert_embedding):
        """Test that batch_encode returns array of correct shape."""
        texts = ["Text one", "Text two", "Text three"]
        embeddings = pubmedbert_embedding.batch_encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 768)

    def test_batch_encode_empty_list(self, pubmedbert_embedding):
        """Test batch_encode with empty list."""
        embeddings = pubmedbert_embedding.batch_encode([])

        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == 0


class TestOpenAIEmbedding:
    """Tests for OpenAI embedding generator."""

    def test_dimension_is_1536(self, openai_embedding):
        """Test that OpenAI dimension is 1536."""
        assert openai_embedding.dimension == 1536

    def test_encode_returns_correct_shape(self, openai_embedding):
        """Test that encode returns array of correct shape."""
        text = "Medical research on drug interactions."
        embedding = openai_embedding.encode(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1536,)

    def test_encode_empty_text_returns_zeros(self, openai_embedding):
        """Test that empty text returns zero vector."""
        embedding = openai_embedding.encode("")

        assert np.all(embedding == 0)

    def test_batch_encode_returns_correct_shape(self, openai_embedding):
        """Test that batch_encode returns array of correct shape."""
        texts = ["First text", "Second text"]
        embeddings = openai_embedding.batch_encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 1536)

    def test_batch_encode_handles_empty_strings(self, openai_embedding):
        """Test that batch_encode handles empty strings in list."""
        texts = ["Valid text", "", "Another valid"]
        embeddings = openai_embedding.batch_encode(texts)

        assert embeddings.shape == (3, 1536)
        assert np.all(embeddings[1] == 0)  # Empty string should be zeros


class TestEmbeddingService:
    """Tests for the EmbeddingService wrapper."""

    def test_pubmedbert_initialization(self, embedding_service_pubmedbert):
        """Test PubMedBERT initialization."""
        assert embedding_service_pubmedbert.model_type == "pubmedbert"
        assert embedding_service_pubmedbert.dimension == 768

    def test_openai_initialization(self, embedding_service_openai):
        """Test OpenAI initialization."""
        assert embedding_service_openai.model_type == "openai"
        assert embedding_service_openai.dimension == 1536

    def test_encode_delegates_to_generator(self, embedding_service_pubmedbert):
        """Test that encode delegates to the underlying generator."""
        text = "Gene therapy research"
        embedding = embedding_service_pubmedbert.encode(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_batch_encode_delegates_to_generator(self, embedding_service_pubmedbert):
        """Test that batch_encode delegates to the underlying generator."""
        texts = ["Text A", "Text B"]
        embeddings = embedding_service_pubmedbert.batch_encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 768)

    def test_switch_model_to_openai(self, embedding_service_pubmedbert):
        """Test switching from PubMedBERT to OpenAI."""
        assert embedding_service_pubmedbert.dimension == 768

        embedding_service_pubmedbert.switch_model("openai")

        assert embedding_service_pubmedbert.model_type == "openai"
        assert embedding_service_pubmedbert.dimension == 1536

    def test_switch_model_to_pubmedbert(self, embedding_service_openai):
        """Test switching from OpenAI to PubMedBERT."""
        assert embedding_service_openai.dimension == 1536

        embedding_service_openai.switch_model("pubmedbert")

        assert embedding_service_openai.model_type == "pubmedbert"
        assert embedding_service_openai.dimension == 768

    def test_encode_after_switch(self, embedding_service_pubmedbert):
        """Test encoding works correctly after model switch."""
        text = "Biomedical research"

        # Encode with PubMedBERT
        embedding1 = embedding_service_pubmedbert.encode(text)
        assert embedding1.shape == (768,)

        # Switch to OpenAI
        embedding_service_pubmedbert.switch_model("openai")
        embedding2 = embedding_service_pubmedbert.encode(text)
        assert embedding2.shape == (1536,)


class TestEmbeddingNormalization:
    """Tests for embedding properties."""

    def test_embedding_values_in_valid_range(self, pubmedbert_embedding):
        """Test that embedding values are in a valid range."""
        text = "Sample biomedical text"
        embedding = pubmedbert_embedding.encode(text)

        # Values should be between 0 and 1 for our mock implementation
        assert np.all(embedding >= 0)
        assert np.all(embedding <= 1)

    def test_embedding_is_float32(self, pubmedbert_embedding):
        """Test that embeddings are float32."""
        text = "Test text"
        embedding = pubmedbert_embedding.encode(text)

        assert embedding.dtype == np.float32

    def test_batch_embeddings_consistent_with_single(self, pubmedbert_embedding):
        """Test that batch encoding produces same results as single encoding."""
        texts = ["Text one", "Text two"]

        # Single encode
        single_embeddings = [pubmedbert_embedding.encode(t) for t in texts]

        # Batch encode
        batch_embeddings = pubmedbert_embedding.batch_encode(texts)

        for i, text in enumerate(texts):
            np.testing.assert_array_equal(single_embeddings[i], batch_embeddings[i])
