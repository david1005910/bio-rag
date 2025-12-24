"""Tests for low coverage modules to reach 80% coverage"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# =============================================================================
# Tests for app/services/search/service.py
# =============================================================================


class TestSearchService:
    """Tests for SearchService"""

    def test_search_service_import(self):
        """Test search service can be imported"""
        from app.services.search.service import SearchService
        assert SearchService is not None

    def test_search_config_dataclass(self):
        """Test SearchConfig dataclass"""
        from app.services.search.service import SearchConfig

        config = SearchConfig()
        assert config.use_hybrid is True
        assert config.use_reranking is True
        assert config.initial_candidates == 50
        assert config.final_results == 10

    def test_search_config_custom(self):
        """Test SearchConfig with custom values"""
        from app.services.search.service import SearchConfig

        config = SearchConfig(
            use_hybrid=False,
            use_reranking=False,
            initial_candidates=100,
            final_results=20,
        )
        assert config.use_hybrid is False
        assert config.final_results == 20

    def test_search_service_init(self):
        """Test search service initialization"""
        from app.services.search.service import SearchService, SearchConfig

        mock_db = MagicMock()
        service = SearchService(db=mock_db)
        assert service.db is mock_db
        assert service._hybrid_searcher is None
        assert service._reranker is None

    def test_search_service_with_config(self):
        """Test search service with custom config"""
        from app.services.search.service import SearchService, SearchConfig

        mock_db = MagicMock()
        config = SearchConfig(use_hybrid=False)
        service = SearchService(db=mock_db, config=config)
        assert service.config.use_hybrid is False


# =============================================================================
# Tests for app/services/rag/chain.py
# =============================================================================


class TestRAGChain:
    """Tests for RAGChain"""

    def test_chain_import(self):
        """Test chain can be imported"""
        from app.services.rag.chain import RAGChain
        assert RAGChain is not None

    def test_chain_init_mocked(self):
        """Test chain initialization with mocks"""
        from app.services.rag.chain import RAGChain

        with patch("app.services.rag.chain.RAGRetriever"):
            with patch("app.services.rag.chain.LLMService"):
                chain = RAGChain()
                assert chain is not None


# =============================================================================
# Tests for app/services/vector/store.py
# =============================================================================


class TestVectorStoreModule:
    """Tests for VectorStore module"""

    def test_search_result_dataclass(self):
        """Test SearchResult dataclass"""
        from app.services.vector.store import SearchResult

        result = SearchResult(
            chunk_id="test_chunk",
            content="Test content",
            score=0.95,
            metadata={"pmid": "12345"},
        )
        assert result.chunk_id == "test_chunk"
        assert result.score == 0.95
        assert result.metadata["pmid"] == "12345"

    def test_vector_store_class(self):
        """Test VectorStore class exists"""
        from app.services.vector.store import VectorStore
        assert VectorStore is not None


# =============================================================================
# Tests for app/services/auth/service.py
# =============================================================================


class TestAuthServiceModule:
    """Tests for AuthService module"""

    def test_authentication_error_class(self):
        """Test AuthenticationError class"""
        from app.services.auth.service import AuthenticationError

        error = AuthenticationError("Invalid credentials")
        assert str(error) == "Invalid credentials"
        assert isinstance(error, Exception)

    def test_auth_service_import(self):
        """Test auth service can be imported"""
        from app.services.auth.service import AuthService
        assert AuthService is not None


# =============================================================================
# Tests for app/services/chat/service.py
# =============================================================================


class TestChatServiceModule:
    """Tests for ChatService module"""

    def test_chat_service_import(self):
        """Test chat service can be imported"""
        from app.services.chat.service import ChatService
        assert ChatService is not None

    def test_chat_service_detect_language(self):
        """Test language detection"""
        from app.services.chat.service import ChatService

        mock_session = MagicMock()
        service = ChatService(mock_session)

        # English detection
        assert service._detect_language("Hello world") == "en"

        # Korean detection
        assert service._detect_language("안녕하세요") == "ko"

        # Mixed - should detect based on Korean threshold
        assert service._detect_language("Hello 세계") in ["en", "ko"]


# =============================================================================
# Tests for app/services/rag/summarizer.py
# =============================================================================


class TestSummarizerModule:
    """Tests for Summarizer module"""

    def test_paper_summarizer_import(self):
        """Test PaperSummarizer can be imported"""
        from app.services.rag.summarizer import PaperSummarizer
        assert PaperSummarizer is not None

    def test_paper_summary_model(self):
        """Test PaperSummary model"""
        from app.services.rag.summarizer import PaperSummary

        summary = PaperSummary(
            paper_id="12345",
            title="Test Paper",
            summary="This paper describes...",
            language="en",
            success=True,
        )
        assert summary.paper_id == "12345"
        assert summary.success is True

    def test_paper_summary_with_error(self):
        """Test PaperSummary with error"""
        from app.services.rag.summarizer import PaperSummary

        summary = PaperSummary(
            paper_id="12345",
            title="Test Paper",
            summary="",
            language="en",
            success=False,
            error="API rate limit exceeded",
        )
        assert summary.success is False
        assert summary.error == "API rate limit exceeded"

    def test_summarizer_system_prompts(self):
        """Test system prompts are defined"""
        from app.services.rag.summarizer import PaperSummarizer

        assert "en" in PaperSummarizer.SYSTEM_PROMPTS
        assert "ko" in PaperSummarizer.SYSTEM_PROMPTS
        assert len(PaperSummarizer.SYSTEM_PROMPTS["en"]) > 0
        assert len(PaperSummarizer.SYSTEM_PROMPTS["ko"]) > 0

    def test_summarizer_language_setting(self):
        """Test summarizer with different languages"""
        from app.services.rag.summarizer import PaperSummarizer

        summarizer_en = PaperSummarizer(language="en")
        assert summarizer_en.language == "en"

        summarizer_ko = PaperSummarizer(language="ko")
        assert summarizer_ko.language == "ko"

    def test_summarizer_with_api_key(self):
        """Test summarizer with API key"""
        from app.services.rag.summarizer import PaperSummarizer

        summarizer = PaperSummarizer(api_key="test-key")
        assert summarizer.api_key == "test-key"


# =============================================================================
# Tests for app/services/rag/retriever.py
# =============================================================================


class TestRetrieverModule:
    """Tests for Retriever module"""

    def test_retrieved_document_dataclass(self):
        """Test RetrievedDocument dataclass"""
        from app.services.rag.retriever import RetrievedDocument

        doc = RetrievedDocument(
            chunk_id="chunk_1",
            pmid="12345",
            title="Test Paper",
            content="Test content",
            section="abstract",
            score=0.9,
            metadata={"year": 2024},
        )
        assert doc.chunk_id == "chunk_1"
        assert doc.pmid == "12345"
        assert doc.score == 0.9

    def test_rag_retriever_singleton(self):
        """Test RAG retriever singleton exists"""
        from app.services.rag.retriever import rag_retriever
        assert rag_retriever is not None


# =============================================================================
# Tests for app/services/rag/llm.py
# =============================================================================


class TestLLMModule:
    """Tests for LLM module"""

    def test_llm_service_import(self):
        """Test LLMService can be imported"""
        from app.services.rag.llm import LLMService
        assert LLMService is not None

    def test_llm_service_init_defaults(self):
        """Test LLMService default initialization"""
        from app.services.rag.llm import LLMService

        service = LLMService()
        assert service.temperature == 0.3
        assert service.max_tokens == 1000

    def test_llm_service_init_custom(self):
        """Test LLMService custom initialization"""
        from app.services.rag.llm import LLMService

        service = LLMService(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=2000,
        )
        assert service.model == "gpt-4o"
        assert service.temperature == 0.7
        assert service.max_tokens == 2000

    def test_llm_service_count_tokens(self):
        """Test token counting"""
        from app.services.rag.llm import LLMService

        service = LLMService()

        # Empty text
        assert service.count_tokens("") == 0

        # Normal text
        text = "This is a test sentence."
        count = service.count_tokens(text)
        assert count > 0
        # Uses len(text) // 4 as approximation
        assert count == len(text) // 4

    def test_llm_singleton(self):
        """Test LLM singleton exists"""
        from app.services.rag.llm import llm_service
        assert llm_service is not None


# =============================================================================
# Tests for app/services/document/extractor.py
# =============================================================================


class TestDocumentExtractorModule:
    """Tests for DocumentExtractor module"""

    def test_document_content_model(self):
        """Test DocumentContent model"""
        from app.services.document.extractor import DocumentContent

        content = DocumentContent(
            source="pdf",
            filepath="/path/to/file.pdf",
            text="Extracted text content",
            text_length=21,
            success=True,
        )
        assert content.source == "pdf"
        assert content.success is True

    def test_document_content_failed(self):
        """Test DocumentContent with failure"""
        from app.services.document.extractor import DocumentContent

        content = DocumentContent(
            source="pdf",
            filepath="/path/to/file.pdf",
            text="",
            text_length=0,
            success=False,
            error="File not found",
        )
        assert content.success is False
        assert content.error == "File not found"

    def test_pdf_downloader_init(self, tmp_path):
        """Test PDFDownloader initialization"""
        from app.services.document.extractor import PDFDownloader

        downloader = PDFDownloader(save_dir=str(tmp_path))
        assert downloader.save_dir.exists()

    def test_pdf_downloader_safe_filename(self, tmp_path):
        """Test safe filename generation"""
        from app.services.document.extractor import PDFDownloader

        downloader = PDFDownloader(save_dir=str(tmp_path))

        # Test special characters removal
        filename = downloader._safe_filename("Title: With <Special> Chars!", "12345")
        assert ":" not in filename
        assert "<" not in filename
        assert ">" not in filename
        assert "!" not in filename

    def test_pdf_downloader_safe_filename_max_length(self, tmp_path):
        """Test safe filename max length"""
        from app.services.document.extractor import PDFDownloader

        downloader = PDFDownloader(save_dir=str(tmp_path))

        long_title = "A" * 200
        filename = downloader._safe_filename(long_title, "12345", max_length=50)
        assert len(filename) <= 60  # max_length + some buffer for id


# =============================================================================
# Tests for app/tasks/celery_app.py
# =============================================================================


class TestCeleryApp:
    """Tests for Celery app"""

    def test_celery_app_import(self):
        """Test Celery app can be imported"""
        from app.tasks.celery_app import celery_app
        assert celery_app is not None
        assert celery_app.main == "bio-rag"


# =============================================================================
# Tests for app/tasks/crawler.py
# =============================================================================


class TestCrawlerTasks:
    """Tests for crawler tasks"""

    def test_crawler_module_import(self):
        """Test crawler module can be imported"""
        from app.tasks import crawler
        assert crawler is not None


# =============================================================================
# Tests for app/tasks/embedding.py
# =============================================================================


class TestEmbeddingTasks:
    """Tests for embedding tasks"""

    def test_embedding_module_import(self):
        """Test embedding module can be imported"""
        from app.tasks import embedding
        assert embedding is not None


# =============================================================================
# Tests for app/core/i18n.py additional tests
# =============================================================================


class TestI18nModuleExtended:
    """Extended tests for i18n module"""

    def test_detect_language_english(self):
        """Test English detection"""
        from app.core.i18n import detect_language

        assert detect_language("Hello world") == "en"
        assert detect_language("This is a test") == "en"
        assert detect_language("Cancer treatment research") == "en"

    def test_detect_language_korean(self):
        """Test Korean detection"""
        from app.core.i18n import detect_language

        assert detect_language("안녕하세요") == "ko"
        assert detect_language("암 치료 연구") == "ko"

    def test_detect_language_empty(self):
        """Test empty string detection"""
        from app.core.i18n import detect_language

        assert detect_language("") == "en"  # Default to English

    def test_detect_language_numbers_only(self):
        """Test numbers only detection"""
        from app.core.i18n import detect_language

        assert detect_language("12345") == "en"  # Default to English

    def test_medical_terms_mapping(self):
        """Test medical terms mapping"""
        from app.core.i18n import MEDICAL_TERMS_KO_EN

        assert "암" in MEDICAL_TERMS_KO_EN
        assert "당뇨" in MEDICAL_TERMS_KO_EN

    def test_translate_medical_terms(self):
        """Test medical term translation"""
        from app.core.i18n import translate_medical_terms

        # Korean to English
        result = translate_medical_terms("폐암", "ko_to_en")
        assert isinstance(result, str)

        # English to Korean
        result = translate_medical_terms("cancer", "en_to_ko")
        assert isinstance(result, str)


# =============================================================================
# Tests for app/api/deps.py
# =============================================================================


class TestAPIDeps:
    """Tests for API dependencies"""

    def test_deps_import(self):
        """Test dependencies can be imported"""
        from app.api import deps
        assert deps is not None


# =============================================================================
# Tests for app/db/session.py
# =============================================================================


class TestDBSession:
    """Tests for database session"""

    def test_session_import(self):
        """Test session can be imported"""
        from app.db import session
        assert session is not None


# =============================================================================
# Tests for app/main.py
# =============================================================================


class TestMainApp:
    """Tests for main application"""

    def test_app_import(self):
        """Test app can be imported"""
        from app.main import app
        assert app is not None
