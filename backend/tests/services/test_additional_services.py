"""Additional service tests for coverage improvement"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.analytics.trend import TrendAnalyzer, TrendData
from app.services.document.extractor import PDFDownloader, DocumentContent
from app.services.rag.chain import RAGChain
from app.services.rag.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_KO,
    USER_TEMPLATE,
    CONTEXT_TEMPLATE,
    format_context,
    build_prompt,
)
from app.services.rag.summarizer import PaperSummarizer, PaperSummary


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return TrendAnalyzer()

    @pytest.fixture
    def sample_papers(self):
        """Sample paper data for testing"""
        return [
            {
                "title": "Cancer immunotherapy advances",
                "abstract": "This study explores immunotherapy for cancer treatment",
                "publication_date": datetime(2024, 1, 15),
            },
            {
                "title": "New drug targets in oncology",
                "abstract": "We identify novel targets for cancer therapy",
                "publication_date": datetime(2023, 6, 20),
            },
            {
                "title": "Machine learning in drug discovery",
                "abstract": "AI approaches for identifying new therapeutic compounds",
                "publication_date": datetime(2024, 3, 10),
            },
        ]

    def test_init(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.papers == []

    def test_init_with_api_key(self):
        """Test analyzer with OpenAI API key"""
        analyzer = TrendAnalyzer(openai_api_key="test-key")
        assert analyzer.openai_api_key == "test-key"

    def test_set_papers(self, analyzer, sample_papers):
        """Test setting papers"""
        analyzer.set_papers(sample_papers)
        assert len(analyzer.papers) == 3

    def test_analyze_publication_trend(self, analyzer, sample_papers):
        """Test analyzing publication trends"""
        analyzer.set_papers(sample_papers)
        analyzer.analyze_publication_trend()

        trend_data = analyzer.trend_data
        assert "year_trend" in trend_data
        assert "years" in trend_data["year_trend"]
        assert "counts" in trend_data["year_trend"]

    def test_extract_key_terms(self, analyzer, sample_papers):
        """Test extracting key terms"""
        analyzer.set_papers(sample_papers)
        analyzer.extract_key_terms(top_n=5)

        trend_data = analyzer.trend_data
        assert "key_terms" in trend_data

    def test_generate_report(self, analyzer, sample_papers):
        """Test generating report"""
        analyzer.set_papers(sample_papers)
        analyzer.analyze_publication_trend()
        analyzer.extract_key_terms()

        report = analyzer.generate_report("cancer")
        assert isinstance(report, str)
        assert len(report) > 0

    @pytest.mark.asyncio
    async def test_analyze(self, analyzer, sample_papers):
        """Test full analysis"""
        analyzer.set_papers(sample_papers)
        result = await analyzer.analyze("cancer")

        assert isinstance(result, TrendData)
        assert result.total_papers == 3


class TestPDFDownloader:
    """Tests for PDFDownloader"""

    @pytest.fixture
    def downloader(self, tmp_path):
        """Create downloader instance with temp directory"""
        return PDFDownloader(save_dir=str(tmp_path))

    def test_init(self, downloader):
        """Test downloader initialization"""
        assert downloader is not None
        assert downloader.save_dir.exists()

    def test_safe_filename(self, downloader):
        """Test safe filename generation"""
        title = "A Test Paper: With Special Characters!"
        paper_id = "12345"
        filename = downloader._safe_filename(title, paper_id)
        assert "12345" in filename
        assert ":" not in filename
        assert "!" not in filename

    def test_safe_filename_long_title(self, downloader):
        """Test safe filename with very long title"""
        title = "A" * 200
        filename = downloader._safe_filename(title, "test123", max_length=50)
        assert len(filename) <= 60  # max_length + id + underscore


class TestDocumentContent:
    """Tests for DocumentContent model"""

    def test_create_document_content(self):
        """Test creating DocumentContent"""
        content = DocumentContent(
            source="pdf",
            filepath="/path/to/file.pdf",
            text="Extracted text content",
            text_length=21,
            success=True,
        )
        assert content.source == "pdf"
        assert content.text_length == 21

    def test_create_failed_document_content(self):
        """Test creating failed DocumentContent"""
        content = DocumentContent(
            source="pdf",
            filepath="/path/to/file.pdf",
            text="",
            text_length=0,
            success=False,
            error="Failed to extract",
        )
        assert not content.success
        assert content.error == "Failed to extract"


class TestPaperSummarizer:
    """Tests for PaperSummarizer"""

    @pytest.fixture
    def summarizer(self):
        """Create summarizer without API key"""
        return PaperSummarizer()

    def test_init(self, summarizer):
        """Test summarizer initialization"""
        assert summarizer is not None
        assert summarizer.api_key is None
        assert summarizer.language == "en"

    def test_init_with_api_key(self):
        """Test summarizer with API key"""
        summarizer = PaperSummarizer(api_key="test-key", language="ko")
        assert summarizer.api_key == "test-key"
        assert summarizer.language == "ko"

    def test_system_prompts_exist(self):
        """Test that system prompts are defined"""
        assert "en" in PaperSummarizer.SYSTEM_PROMPTS
        assert "ko" in PaperSummarizer.SYSTEM_PROMPTS


class TestPaperSummaryModel:
    """Tests for PaperSummary model"""

    def test_create_paper_summary(self):
        """Test creating PaperSummary"""
        summary = PaperSummary(
            paper_id="12345",
            title="Test Paper",
            summary="This paper describes...",
            language="en",
            success=True,
        )
        assert summary.paper_id == "12345"
        assert summary.success

    def test_create_failed_summary(self):
        """Test creating failed PaperSummary"""
        summary = PaperSummary(
            paper_id="12345",
            title="Test Paper",
            summary="",
            language="en",
            success=False,
            error="API error",
        )
        assert not summary.success
        assert summary.error == "API error"


class TestRAGPrompts:
    """Tests for RAG prompts"""

    def test_english_system_prompt_exists(self):
        """Test that English system prompt is defined"""
        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 0
        assert "biomedical" in SYSTEM_PROMPT.lower()

    def test_korean_system_prompt_exists(self):
        """Test that Korean system prompt is defined"""
        assert SYSTEM_PROMPT_KO is not None
        assert len(SYSTEM_PROMPT_KO) > 0

    def test_user_template_exists(self):
        """Test that user template is defined"""
        assert USER_TEMPLATE is not None
        assert "{context}" in USER_TEMPLATE
        assert "{question}" in USER_TEMPLATE

    def test_context_template_exists(self):
        """Test that context template is defined"""
        assert CONTEXT_TEMPLATE is not None
        assert "{pmid}" in CONTEXT_TEMPLATE

    def test_format_context(self):
        """Test context formatting function"""
        chunks = [
            {"pmid": "12345", "title": "Test Paper", "section": "abstract", "content": "Test content"},
            {"pmid": "67890", "title": "Another Paper", "section": "methods", "content": "More content"},
        ]
        result = format_context(chunks)
        assert "12345" in result
        assert "67890" in result
        assert "Test Paper" in result

    def test_format_context_empty(self):
        """Test formatting empty context"""
        result = format_context([])
        assert result == ""

    def test_build_prompt_english(self):
        """Test building English prompt"""
        system, user = build_prompt("What is cancer?", "Context here", language="en")
        assert "biomedical" in system.lower()
        assert "What is cancer?" in user

    def test_build_prompt_korean(self):
        """Test building Korean prompt"""
        system, user = build_prompt("암이란 무엇인가요?", "컨텍스트", language="ko")
        assert "바이오의학" in system or "연구" in system


class TestRAGChain:
    """Tests for RAGChain"""

    @pytest.fixture
    def chain(self):
        """Create RAG chain without dependencies"""
        with patch("app.services.rag.chain.RAGRetriever"):
            with patch("app.services.rag.chain.LLMService"):
                return RAGChain()

    def test_init(self, chain):
        """Test chain initialization"""
        assert chain is not None


class TestDemoService:
    """Tests for demo service"""

    def test_get_demo_search_results(self):
        """Test getting demo search results"""
        from app.services.demo import get_demo_search_results

        result = get_demo_search_results("cancer", 5)
        assert "results" in result
        assert "total" in result
        assert len(result["results"]) <= 5

    def test_get_demo_chat_response(self):
        """Test getting demo chat response"""
        from app.services.demo import get_demo_chat_response

        result = get_demo_chat_response("What is cancer?")
        assert "answer" in result
        assert "citations" in result


class TestAuthService:
    """Tests for authentication service"""

    def test_auth_error_class(self):
        """Test AuthenticationError class"""
        from app.services.auth.service import AuthenticationError

        error = AuthenticationError("Test error")
        assert str(error) == "Test error"

    def test_security_functions(self):
        """Test security helper functions"""
        from app.services.auth.security import (
            hash_password,
            verify_password,
            create_access_token,
            create_refresh_token,
        )

        # Test password hashing
        password = "testpassword123"
        hashed = hash_password(password)
        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrongpassword", hashed)

        # Test token creation
        user_id = "test-user-id"
        access_token = create_access_token(user_id)
        refresh_token = create_refresh_token(user_id)
        assert access_token is not None
        assert refresh_token is not None
        assert access_token != refresh_token


class TestChatService:
    """Tests for chat service"""

    def test_detect_language(self):
        """Test language detection in chat service"""
        from app.services.chat.service import ChatService

        mock_session = MagicMock()
        service = ChatService(mock_session)

        assert service._detect_language("Hello world") == "en"
        assert service._detect_language("안녕하세요") == "ko"


class TestVectorStore:
    """Tests for vector store"""

    def test_search_result_creation(self):
        """Test SearchResult dataclass"""
        from app.services.vector.store import SearchResult

        result = SearchResult(
            chunk_id="test_123",
            content="Test content",
            score=0.9,
            metadata={"key": "value"},
        )
        assert result.chunk_id == "test_123"
        assert result.score == 0.9


class TestLLMService:
    """Tests for LLM service"""

    def test_init(self):
        """Test LLM service initialization"""
        from app.services.rag.llm import LLMService

        service = LLMService()
        assert service is not None
        assert service.temperature == 0.3
        assert service.max_tokens == 1000

    def test_init_with_custom_params(self):
        """Test LLM service with custom parameters"""
        from app.services.rag.llm import LLMService

        service = LLMService(model="gpt-4", temperature=0.7, max_tokens=2000)
        assert service.model == "gpt-4"
        assert service.temperature == 0.7
        assert service.max_tokens == 2000

    def test_count_tokens(self):
        """Test token counting"""
        from app.services.rag.llm import LLMService

        service = LLMService()
        text = "This is a test string with some words."
        token_count = service.count_tokens(text)
        assert token_count > 0
        assert token_count == len(text) // 4


class TestReranker:
    """Tests for reranker service"""

    def test_import(self):
        """Test that reranker can be imported"""
        from app.services.rag.reranker import CrossEncoderReranker

        assert CrossEncoderReranker is not None


class TestRAGRetriever:
    """Tests for RAG retriever"""

    def test_retrieved_document_dataclass(self):
        """Test RetrievedDocument dataclass"""
        from app.services.rag.retriever import RetrievedDocument

        doc = RetrievedDocument(
            chunk_id="test_0",
            pmid="12345",
            title="Test Paper",
            content="Test content",
            section="abstract",
            score=0.9,
            metadata={"key": "value"},
        )
        assert doc.pmid == "12345"
        assert doc.score == 0.9
