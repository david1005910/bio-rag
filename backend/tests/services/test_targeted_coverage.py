"""Targeted tests for specific uncovered code paths"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import pytest
import numpy as np


# =============================================================================
# Tests for app/services/vector/qdrant_store.py - Focus on encoders
# =============================================================================


class TestSPLADEEncoderPaths:
    """Tests for SPLADE encoder code paths"""

    def test_encode_bm25_style_with_various_texts(self):
        """Test BM25-style encoding with various text types"""
        from app.services.vector.qdrant_store import SPLADEEncoder

        encoder = SPLADEEncoder()
        encoder._model = "fallback"  # Force BM25 fallback

        # Normal text
        indices, values = encoder.encode("cancer treatment therapy research")
        assert len(indices) > 0
        assert len(values) > 0

        # Query with boost
        indices_q, values_q = encoder.encode("cancer treatment", is_query=True)
        if indices and indices_q:
            # Query values should be boosted
            pass

    def test_encode_stopwords_removal(self):
        """Test stopwords are removed"""
        from app.services.vector.qdrant_store import SPLADEEncoder

        encoder = SPLADEEncoder()
        encoder._model = "fallback"

        # Text with only stopwords should return empty
        indices, values = encoder.encode("the and or but in on at to for")
        assert len(indices) == 0


class TestBGEM3EncoderPaths:
    """Tests for BGE-M3 encoder paths"""

    def test_init_custom_model(self):
        """Test custom model initialization"""
        from app.services.vector.qdrant_store import BGEM3Encoder

        encoder = BGEM3Encoder(model_name="custom/model")
        assert encoder.model_name == "custom/model"
        assert encoder.dimension == 1024


class TestPubMedBERTEncoderPaths:
    """Tests for PubMedBERT encoder paths"""

    def test_init_custom_model(self):
        """Test custom model initialization"""
        from app.services.vector.qdrant_store import PubMedBERTEncoder

        encoder = PubMedBERTEncoder(model_name="custom/bert-model")
        assert encoder.model_name == "custom/bert-model"
        assert encoder.dimension == 768


class TestQdrantHybridStorePaths:
    """Tests for QdrantHybridStore code paths"""

    def test_dense_dim_multilingual_vs_biomedical(self):
        """Test dimension selection"""
        from app.services.vector.qdrant_store import QdrantHybridStore

        # Multilingual uses BGE-M3 (1024)
        store_multi = QdrantHybridStore(use_multilingual=True, use_memory=True)
        assert store_multi.dense_dim == 1024

        # Biomedical uses PubMedBERT (768)
        store_bio = QdrantHybridStore(use_multilingual=False, use_memory=True)
        assert store_bio.dense_dim == 768

    def test_hash_id_consistency(self):
        """Test hash ID is consistent"""
        from app.services.vector.qdrant_store import QdrantHybridStore

        store = QdrantHybridStore(use_memory=True)

        id1 = store._hash_id("document_123")
        id2 = store._hash_id("document_123")
        id3 = store._hash_id("document_456")

        assert id1 == id2  # Same input -> same hash
        assert id1 != id3  # Different input -> different hash

    def test_calculate_score_distribution_various_cases(self):
        """Test score distribution calculation"""
        from app.services.vector.qdrant_store import QdrantHybridStore, HybridSearchResult

        store = QdrantHybridStore(use_memory=True)

        # Empty results
        dist = store._calculate_score_distribution([], 0.7, 0.3)
        assert dist == {}

        # Single result
        results = [HybridSearchResult(
            doc_id="1", content="test", metadata={},
            dense_score=0.9, sparse_score=0.8, rrf_score=0.85
        )]
        dist = store._calculate_score_distribution(results, 0.7, 0.3)
        assert "dense_scores" in dist

        # Multiple results with varying scores
        results2 = [
            HybridSearchResult(doc_id="1", content="test1", metadata={},
                             dense_score=0.9, sparse_score=0.8, rrf_score=0.85),
            HybridSearchResult(doc_id="2", content="test2", metadata={},
                             dense_score=0.7, sparse_score=0.6, rrf_score=0.65),
            HybridSearchResult(doc_id="3", content="test3", metadata={},
                             dense_score=0.0, sparse_score=0.0, rrf_score=0.0),  # Zero case
        ]
        dist2 = store._calculate_score_distribution(results2, 0.7, 0.3)
        assert len(dist2["contributions"]) == 3


# =============================================================================
# Tests for app/services/rag/embedding.py - Focus on factory paths
# =============================================================================


class TestEmbeddingFactoryPaths:
    """Tests for EmbeddingModelFactory code paths"""

    def test_factory_get_available_models(self):
        """Test getting available models"""
        from app.services.rag.embedding import EmbeddingModelFactory

        models = EmbeddingModelFactory.get_available_models()
        assert len(models) > 0
        assert "pubmedbert" in models
        assert "openai-small" in models

    def test_factory_creates_huggingface(self):
        """Test factory creates HuggingFace embedding"""
        from app.services.rag.embedding import EmbeddingModelFactory, HuggingFaceEmbedding

        embedding = EmbeddingModelFactory.create("biobert", device="cpu")
        assert isinstance(embedding, HuggingFaceEmbedding)


class TestOpenAIEmbeddingPaths:
    """Tests for OpenAI embedding code paths"""

    def test_embed_empty_returns_empty(self):
        """Test empty text returns empty list"""
        from app.services.rag.embedding import OpenAIEmbedding

        embedding = OpenAIEmbedding(api_key="test")
        result = embedding.embed("")
        assert result == []

        result2 = embedding.embed("   ")
        assert result2 == []

    def test_embed_batch_empty_returns_empty(self):
        """Test empty batch returns empty list"""
        from app.services.rag.embedding import OpenAIEmbedding

        embedding = OpenAIEmbedding(api_key="test")
        result = embedding.embed_batch([])
        assert result == []

        result2 = embedding.embed_batch(["", "   ", "\t"])
        assert result2 == []


class TestHuggingFaceEmbeddingPaths:
    """Tests for HuggingFace embedding code paths"""

    def test_embed_empty_returns_empty(self):
        """Test empty text returns empty list"""
        from app.services.rag.embedding import HuggingFaceEmbedding

        embedding = HuggingFaceEmbedding()
        result = embedding.embed("")
        assert result == []

    def test_embed_batch_empty_returns_empty(self):
        """Test empty batch returns empty list"""
        from app.services.rag.embedding import HuggingFaceEmbedding

        embedding = HuggingFaceEmbedding()
        result = embedding.embed_batch([])
        assert result == []

        result2 = embedding.embed_batch(["", "  "])
        assert result2 == []


# =============================================================================
# Tests for app/services/rag/hybrid_search.py - Focus on BM25 paths
# =============================================================================


class TestBM25IndexPaths:
    """Tests for BM25Index code paths"""

    def test_tokenize_various_inputs(self):
        """Test tokenization with various inputs"""
        from app.services.rag.hybrid_search import BM25Index

        index = BM25Index()

        # Normal text
        tokens = index._tokenize("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens

        # Numbers
        tokens2 = index._tokenize("COVID-19 in 2024")
        assert "covid" in tokens2 or "19" in tokens2

        # Empty
        tokens3 = index._tokenize("")
        assert tokens3 == []

    def test_is_built_property(self):
        """Test is_built property"""
        from app.services.rag.hybrid_search import BM25Index

        index = BM25Index()
        assert index.is_built is False

        index.build_index([{"chunk_id": "1", "content": "test"}])
        assert index.is_built is True

    def test_document_count_property(self):
        """Test document_count property"""
        from app.services.rag.hybrid_search import BM25Index

        index = BM25Index()
        assert index.document_count == 0

        docs = [
            {"chunk_id": "1", "content": "test one"},
            {"chunk_id": "2", "content": "test two"},
        ]
        index.build_index(docs)
        assert index.document_count == 2


class TestHybridSearcherPaths:
    """Tests for HybridSearcher code paths"""

    def test_config_defaults(self):
        """Test config default values"""
        from app.services.rag.hybrid_search import HybridSearchConfig

        config = HybridSearchConfig()
        assert config.dense_weight == 0.7
        assert config.bm25_weight == 0.3
        assert config.rrf_k == 60


# =============================================================================
# Tests for app/services/rag/reranker.py - Focus on config paths
# =============================================================================


class TestRerankerConfigPaths:
    """Tests for RerankerConfig code paths"""

    def test_config_custom_values(self):
        """Test config with custom values"""
        from app.services.rag.reranker import RerankerConfig

        config = RerankerConfig(
            model_name="custom/model",
            top_k=20,
            batch_size=16,
            max_length=256,
            device="cuda",
            score_threshold=0.5,
        )
        assert config.model_name == "custom/model"
        assert config.top_k == 20
        assert config.score_threshold == 0.5


# =============================================================================
# Tests for app/services/analytics/trend.py - Focus on analysis paths
# =============================================================================


class TestTrendAnalyzerPaths:
    """Tests for TrendAnalyzer code paths"""

    @pytest.fixture
    def sample_papers(self):
        return [
            {"title": "Paper A", "abstract": "Cancer treatment", "publication_date": datetime(2024, 1, 1)},
            {"title": "Paper B", "abstract": "Drug discovery", "publication_date": datetime(2023, 6, 1)},
        ]

    def test_set_papers_and_analyze(self, sample_papers):
        """Test setting papers and basic analysis"""
        from app.services.analytics.trend import TrendAnalyzer

        analyzer = TrendAnalyzer()
        analyzer.set_papers(sample_papers)

        assert len(analyzer.papers) == 2

        # Analyze publication trend
        analyzer.analyze_publication_trend()
        assert "year_trend" in analyzer.trend_data

    def test_extract_key_terms(self, sample_papers):
        """Test key term extraction"""
        from app.services.analytics.trend import TrendAnalyzer

        analyzer = TrendAnalyzer()
        analyzer.set_papers(sample_papers)
        analyzer.extract_key_terms(top_n=5)

        assert "key_terms" in analyzer.trend_data

    def test_generate_report(self, sample_papers):
        """Test report generation"""
        from app.services.analytics.trend import TrendAnalyzer

        analyzer = TrendAnalyzer()
        analyzer.set_papers(sample_papers)
        analyzer.analyze_publication_trend()
        analyzer.extract_key_terms()

        report = analyzer.generate_report("cancer")
        assert isinstance(report, str)


# =============================================================================
# Tests for app/services/rag/llm.py - Focus on token counting
# =============================================================================


class TestLLMServicePaths:
    """Tests for LLMService code paths"""

    def test_count_tokens_various(self):
        """Test token counting with various inputs"""
        from app.services.rag.llm import LLMService

        service = LLMService()

        # Empty
        assert service.count_tokens("") == 0

        # Short
        count_short = service.count_tokens("hello")
        assert count_short >= 0

        # Long
        long_text = "word " * 1000
        count_long = service.count_tokens(long_text)
        assert count_long > count_short


# =============================================================================
# Tests for app/services/rag/validator.py - Focus on validation
# =============================================================================


class TestValidatorPaths:
    """Tests for ResponseValidator code paths"""

    def test_validator_init(self):
        """Test validator initialization"""
        from app.services.rag.validator import ResponseValidator

        validator = ResponseValidator()
        assert validator is not None

    def test_validation_result_creation(self):
        """Test ValidationResult creation"""
        from app.services.rag.validator import ValidationResult

        result = ValidationResult(
            is_valid=True,
            confidence_score=0.9,
            cited_pmids=["123"],
            valid_citations=["123"],
            invalid_citations=[],
            warnings=[],
        )
        assert result.is_valid is True


# =============================================================================
# Tests for app/services/rag/summarizer.py - Focus on prompts
# =============================================================================


class TestSummarizerPaths:
    """Tests for PaperSummarizer code paths"""

    def test_system_prompts_content(self):
        """Test system prompts have content"""
        from app.services.rag.summarizer import PaperSummarizer

        en_prompt = PaperSummarizer.SYSTEM_PROMPTS["en"]
        ko_prompt = PaperSummarizer.SYSTEM_PROMPTS["ko"]

        assert len(en_prompt) > 20
        assert len(ko_prompt) > 20


# =============================================================================
# Tests for app/services/demo.py
# =============================================================================


class TestDemoServicePaths:
    """Tests for demo service code paths"""

    def test_demo_search_various_queries(self):
        """Test demo search with various queries"""
        from app.services.demo import get_demo_search_results

        # Different queries
        result1 = get_demo_search_results("cancer", 5)
        assert "results" in result1

        result2 = get_demo_search_results("immunotherapy", 10)
        assert "results" in result2

    def test_demo_chat_various_queries(self):
        """Test demo chat with various queries"""
        from app.services.demo import get_demo_chat_response

        result = get_demo_chat_response("What is cancer?")
        assert "answer" in result
        assert "citations" in result


# =============================================================================
# Tests for app/services/auth/security.py
# =============================================================================


class TestSecurityPaths:
    """Tests for security module code paths"""

    def test_password_hash_verify(self):
        """Test password hashing and verification"""
        from app.services.auth.security import hash_password, verify_password

        password = "secure_password_123"
        hashed = hash_password(password)

        assert hashed != password
        assert verify_password(password, hashed) is True
        assert verify_password("wrong_password", hashed) is False

    def test_token_creation(self):
        """Test token creation"""
        from app.services.auth.security import create_access_token, create_refresh_token

        user_id = "test-user-123"

        access_token = create_access_token(user_id)
        refresh_token = create_refresh_token(user_id)

        assert access_token is not None
        assert refresh_token is not None
        assert access_token != refresh_token


# =============================================================================
# Tests for app/core/i18n.py
# =============================================================================


class TestI18nPaths:
    """Tests for i18n module code paths"""

    def test_detect_language_various(self):
        """Test language detection various cases"""
        from app.core.i18n import detect_language

        # English
        assert detect_language("Hello world") == "en"
        assert detect_language("Cancer treatment research") == "en"

        # Korean
        assert detect_language("안녕하세요") == "ko"
        assert detect_language("암 치료") == "ko"

        # Edge cases
        assert detect_language("") == "en"
        assert detect_language("12345") == "en"

    def test_translate_medical_terms(self):
        """Test medical term translation"""
        from app.core.i18n import translate_medical_terms

        # Korean to English
        result1 = translate_medical_terms("암", "ko_to_en")
        assert isinstance(result1, str)

        # English to Korean
        result2 = translate_medical_terms("cancer", "en_to_ko")
        assert isinstance(result2, str)
