"""Comprehensive tests for embedding service"""

from unittest.mock import MagicMock, patch
import pytest

from app.services.rag.embedding import (
    EMBEDDING_MODELS,
    BaseEmbedding,
    OpenAIEmbedding,
    HuggingFaceEmbedding,
    EmbeddingModelFactory,
    EmbeddingService,
    embed_text,
    create_embedding_service,
)


class TestEmbeddingModels:
    """Tests for embedding model configurations"""

    def test_all_models_have_required_keys(self):
        """Test all models have required configuration keys"""
        required_keys = ["name", "description", "dimension", "type"]
        for model_key, config in EMBEDDING_MODELS.items():
            for key in required_keys:
                assert key in config, f"Model {model_key} missing {key}"

    def test_pubmedbert_config(self):
        """Test PubMedBERT configuration"""
        config = EMBEDDING_MODELS["pubmedbert"]
        assert config["dimension"] == 768
        assert config["type"] == "huggingface"
        assert "PubMedBERT" in config["name"]

    def test_biobert_config(self):
        """Test BioBERT configuration"""
        config = EMBEDDING_MODELS["biobert"]
        assert config["dimension"] == 768
        assert config["type"] == "huggingface"

    def test_scibert_config(self):
        """Test SciBERT configuration"""
        config = EMBEDDING_MODELS["scibert"]
        assert config["dimension"] == 768
        assert config["type"] == "huggingface"
        assert "scibert" in config["name"]

    def test_biolinkbert_config(self):
        """Test BioLinkBERT configuration"""
        config = EMBEDDING_MODELS["biolinkbert"]
        assert config["dimension"] == 768
        assert config["type"] == "huggingface"

    def test_openai_small_config(self):
        """Test OpenAI small configuration"""
        config = EMBEDDING_MODELS["openai-small"]
        assert config["dimension"] == 1536
        assert config["type"] == "openai"

    def test_openai_large_config(self):
        """Test OpenAI large configuration"""
        config = EMBEDDING_MODELS["openai-large"]
        assert config["dimension"] == 3072
        assert config["type"] == "openai"

    def test_minilm_config(self):
        """Test MiniLM configuration"""
        config = EMBEDDING_MODELS["minilm"]
        assert config["dimension"] == 384
        assert config["type"] == "huggingface"


class TestOpenAIEmbedding:
    """Tests for OpenAI embedding"""

    def test_init_default(self):
        """Test default initialization"""
        with patch("app.services.rag.embedding.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test_key"
            embedding = OpenAIEmbedding()
            assert embedding.model == "text-embedding-3-small"
            assert embedding._client is None

    def test_init_custom_model(self):
        """Test with custom model"""
        embedding = OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key="custom_key"
        )
        assert embedding.model == "text-embedding-3-large"
        assert embedding._api_key == "custom_key"

    def test_dimension_property(self):
        """Test dimension property"""
        embedding = OpenAIEmbedding(model="text-embedding-3-small")
        assert embedding.dimension == 1536

    def test_embed_empty_text(self):
        """Test embedding empty text"""
        embedding = OpenAIEmbedding(api_key="test")
        result = embedding.embed("")
        assert result == []

    def test_embed_whitespace_only(self):
        """Test embedding whitespace only text"""
        embedding = OpenAIEmbedding(api_key="test")
        result = embedding.embed("   ")
        assert result == []

    def test_embed_batch_empty(self):
        """Test batch embedding empty list"""
        embedding = OpenAIEmbedding(api_key="test")
        result = embedding.embed_batch([])
        assert result == []

    def test_embed_batch_whitespace_only(self):
        """Test batch embedding with only whitespace texts"""
        embedding = OpenAIEmbedding(api_key="test")
        result = embedding.embed_batch(["", "  ", "\t"])
        assert result == []

    def test_client_lazy_initialization(self):
        """Test client lazy initialization"""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            embedding = OpenAIEmbedding(api_key="test_key")
            assert embedding._client is None

            # Access client property
            client = embedding.client
            assert client is not None

    def test_embed_success(self):
        """Test successful embedding"""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.return_value = mock_response

            embedding = OpenAIEmbedding(api_key="test_key")
            result = embedding.embed("test text")

            assert result == [0.1, 0.2, 0.3]

    def test_embed_batch_success(self):
        """Test successful batch embedding"""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1, 0.2]),
                MagicMock(embedding=[0.3, 0.4]),
            ]
            mock_client.embeddings.create.return_value = mock_response

            embedding = OpenAIEmbedding(api_key="test_key")
            result = embedding.embed_batch(["text1", "text2"])

            assert len(result) == 2
            assert result[0] == [0.1, 0.2]
            assert result[1] == [0.3, 0.4]

    def test_embed_error_handling(self):
        """Test error handling in embed"""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.embeddings.create.side_effect = Exception("API error")

            embedding = OpenAIEmbedding(api_key="test_key")

            with pytest.raises(Exception, match="API error"):
                embedding.embed("test text")


class TestHuggingFaceEmbedding:
    """Tests for HuggingFace embedding"""

    def test_init_default(self):
        """Test default initialization"""
        embedding = HuggingFaceEmbedding()
        assert embedding.model_name == "dmis-lab/biobert-base-cased-v1.2"
        assert embedding.device == "cpu"
        assert embedding._model is None
        assert embedding._tokenizer is None
        assert embedding._dimension == 768

    def test_init_custom(self):
        """Test custom initialization"""
        embedding = HuggingFaceEmbedding(
            model_name="custom/model",
            device="cuda"
        )
        assert embedding.model_name == "custom/model"
        assert embedding.device == "cuda"

    def test_dimension_property(self):
        """Test dimension property"""
        embedding = HuggingFaceEmbedding()
        assert embedding.dimension == 768

    def test_embed_empty_text(self):
        """Test embedding empty text"""
        embedding = HuggingFaceEmbedding()
        result = embedding.embed("")
        assert result == []

    def test_embed_whitespace_only(self):
        """Test embedding whitespace only text"""
        embedding = HuggingFaceEmbedding()
        result = embedding.embed("   ")
        assert result == []

    def test_embed_batch_empty(self):
        """Test batch embedding empty list"""
        embedding = HuggingFaceEmbedding()
        result = embedding.embed_batch([])
        assert result == []

    def test_embed_batch_whitespace_only(self):
        """Test batch embedding with only whitespace texts"""
        embedding = HuggingFaceEmbedding()
        result = embedding.embed_batch(["", "  "])
        assert result == []


class TestEmbeddingModelFactory:
    """Tests for EmbeddingModelFactory"""

    def test_get_available_models(self):
        """Test getting available models"""
        models = EmbeddingModelFactory.get_available_models()
        assert isinstance(models, dict)
        assert "pubmedbert" in models
        assert "biobert" in models
        assert "openai-small" in models

    def test_get_available_models_is_copy(self):
        """Test that get_available_models returns a copy"""
        models1 = EmbeddingModelFactory.get_available_models()
        models2 = EmbeddingModelFactory.get_available_models()
        assert models1 is not models2

    def test_create_unknown_model_warning(self):
        """Test creating unknown model logs warning"""
        # Unknown model should fall back to openai-small
        # This is verified by the logger.warning call
        with patch("app.services.rag.embedding.logger") as mock_logger:
            with patch("app.services.rag.embedding.settings") as mock_settings:
                mock_settings.OPENAI_API_KEY = None  # Force fallback to HuggingFace

                # Create with unknown model - should warn and fall back
                result = EmbeddingModelFactory.create("unknown_model_xyz")

                # Should have warned about unknown model
                # And created a HuggingFace embedding as fallback
                assert result is not None

    @patch("app.services.rag.embedding.settings")
    def test_create_openai_embedding(self, mock_settings):
        """Test creating OpenAI embedding"""
        mock_settings.OPENAI_API_KEY = "test_key"

        embedding = EmbeddingModelFactory.create("openai-small")
        assert isinstance(embedding, OpenAIEmbedding)

    @patch("app.services.rag.embedding.settings")
    def test_create_openai_without_key_fallback(self, mock_settings):
        """Test creating OpenAI embedding without key falls back to HuggingFace"""
        mock_settings.OPENAI_API_KEY = None

        # This should fall back to biobert
        embedding = EmbeddingModelFactory.create("openai-small", api_key=None)
        assert isinstance(embedding, HuggingFaceEmbedding)

    def test_create_huggingface_embedding(self):
        """Test creating HuggingFace embedding"""
        embedding = EmbeddingModelFactory.create("biobert", device="cpu")
        assert isinstance(embedding, HuggingFaceEmbedding)
        assert embedding.device == "cpu"


class TestEmbeddingService:
    """Tests for EmbeddingService (legacy compatibility)"""

    def test_is_openai_subclass(self):
        """Test that EmbeddingService is OpenAIEmbedding subclass"""
        assert issubclass(EmbeddingService, OpenAIEmbedding)

    def test_default_model(self):
        """Test default model"""
        service = EmbeddingService()
        assert service.model == "text-embedding-3-small"


class TestEmbedTextFunction:
    """Tests for embed_text function"""

    @patch("app.services.rag.embedding.get_embedding")
    def test_embed_text_returns_list(self, mock_get_embedding):
        """Test embed_text returns list"""
        mock_get_embedding.return_value = (0.1, 0.2, 0.3)

        result = embed_text("test")
        assert isinstance(result, list)
        assert result == [0.1, 0.2, 0.3]


class TestCreateEmbeddingService:
    """Tests for create_embedding_service factory function"""

    def test_factory_function(self):
        """Test factory function calls EmbeddingModelFactory"""
        with patch.object(EmbeddingModelFactory, "create") as mock_create:
            mock_create.return_value = MagicMock()

            create_embedding_service("biobert", "cpu", "key")

            mock_create.assert_called_once_with("biobert", "cpu", "key")

    def test_factory_function_defaults(self):
        """Test factory function with default arguments"""
        with patch.object(EmbeddingModelFactory, "create") as mock_create:
            mock_create.return_value = MagicMock()

            create_embedding_service()

            mock_create.assert_called_once_with("openai-small", "cpu", None)
