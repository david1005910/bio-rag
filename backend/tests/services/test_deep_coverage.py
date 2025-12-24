"""Deep coverage tests for low-coverage modules"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestEmbeddingModelsDeep:
    """Deep tests for embedding models"""

    def test_all_model_configs(self):
        """Test all embedding model configurations"""
        from app.services.rag.embedding import EMBEDDING_MODELS

        required_keys = ["name", "dimension", "type"]
        for model_key, config in EMBEDDING_MODELS.items():
            for key in ["name", "dimension"]:
                assert key in config, f"Missing {key} in {model_key}"

    def test_pubmedbert_config(self):
        """Test PubMedBERT configuration"""
        from app.services.rag.embedding import EMBEDDING_MODELS

        config = EMBEDDING_MODELS["pubmedbert"]
        assert config["dimension"] == 768
        assert "PubMedBERT" in config["name"]

    def test_biobert_config(self):
        """Test BioBERT configuration"""
        from app.services.rag.embedding import EMBEDDING_MODELS

        config = EMBEDDING_MODELS["biobert"]
        assert config["dimension"] == 768

    def test_scibert_config(self):
        """Test SciBERT configuration"""
        from app.services.rag.embedding import EMBEDDING_MODELS

        config = EMBEDDING_MODELS["scibert"]
        assert config["dimension"] == 768


class TestTrendAnalyzerDeep:
    """Deep tests for trend analyzer"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        from app.services.analytics.trend import TrendAnalyzer
        return TrendAnalyzer()

    @pytest.fixture
    def papers_with_dates(self):
        """Sample papers with various dates"""
        return [
            {"title": "Paper 1", "abstract": "Cancer research", "publication_date": datetime(2024, 1, 15)},
            {"title": "Paper 2", "abstract": "Cancer treatment", "publication_date": datetime(2024, 2, 15)},
            {"title": "Paper 3", "abstract": "Immunotherapy cancer", "publication_date": datetime(2023, 6, 15)},
            {"title": "Paper 4", "abstract": "Gene therapy treatment", "publication_date": datetime(2023, 3, 15)},
            {"title": "Paper 5", "abstract": "Drug discovery AI", "publication_date": datetime(2022, 12, 15)},
        ]

    def test_stopwords_defined(self, analyzer):
        """Test stopwords are defined"""
        assert len(analyzer.STOPWORDS) > 0
        assert "the" in analyzer.STOPWORDS
        assert "and" in analyzer.STOPWORDS

    def test_set_papers(self, analyzer, papers_with_dates):
        """Test setting papers"""
        analyzer.set_papers(papers_with_dates)
        assert len(analyzer.papers) == 5

    def test_analyze_publication_trend(self, analyzer, papers_with_dates):
        """Test publication trend analysis"""
        analyzer.set_papers(papers_with_dates)
        analyzer.analyze_publication_trend()

        assert "year_trend" in analyzer.trend_data
        year_trend = analyzer.trend_data["year_trend"]
        assert "years" in year_trend
        assert "counts" in year_trend

    def test_extract_key_terms(self, analyzer, papers_with_dates):
        """Test key term extraction"""
        analyzer.set_papers(papers_with_dates)
        analyzer.extract_key_terms(top_n=10)

        assert "key_terms" in analyzer.trend_data

    def test_generate_report(self, analyzer, papers_with_dates):
        """Test report generation"""
        analyzer.set_papers(papers_with_dates)
        analyzer.analyze_publication_trend()
        analyzer.extract_key_terms()

        report = analyzer.generate_report("cancer")
        assert isinstance(report, str)
        assert len(report) > 0


class TestDocumentExtractorDeep:
    """Deep tests for document extractor"""

    def test_pdf_downloader_save_dir_creation(self, tmp_path):
        """Test PDF downloader creates save directory"""
        from app.services.document.extractor import PDFDownloader

        save_dir = tmp_path / "papers"
        downloader = PDFDownloader(save_dir=str(save_dir))
        assert save_dir.exists()

    def test_safe_filename_special_chars(self, tmp_path):
        """Test safe filename with special characters"""
        from app.services.document.extractor import PDFDownloader

        downloader = PDFDownloader(save_dir=str(tmp_path))

        # Test various special characters
        test_cases = [
            ("Title: With Colon", "12345"),
            ("Title / With Slash", "12345"),
            ("Title <With> Brackets", "12345"),
            ('Title "With" Quotes', "12345"),
        ]

        for title, paper_id in test_cases:
            filename = downloader._safe_filename(title, paper_id)
            assert ":" not in filename
            assert "/" not in filename
            assert "<" not in filename
            assert ">" not in filename
            assert '"' not in filename


class TestSummarizerDeep:
    """Deep tests for summarizer"""

    def test_system_prompts_content(self):
        """Test system prompts have meaningful content"""
        from app.services.rag.summarizer import PaperSummarizer

        assert "research" in PaperSummarizer.SYSTEM_PROMPTS["en"].lower()
        assert len(PaperSummarizer.SYSTEM_PROMPTS["ko"]) > 50

    def test_summarizer_language_setting(self):
        """Test summarizer language setting"""
        from app.services.rag.summarizer import PaperSummarizer

        en_summarizer = PaperSummarizer(language="en")
        assert en_summarizer.language == "en"

        ko_summarizer = PaperSummarizer(language="ko")
        assert ko_summarizer.language == "ko"


class TestSearchServiceDeep:
    """Deep tests for search service"""

    def test_search_service_import(self):
        """Test search service can be imported"""
        from app.services.search.service import SearchService

        assert SearchService is not None


class TestValidatorDeep:
    """Deep tests for response validator"""

    def test_pmid_pattern_matches(self):
        """Test PMID pattern matching"""
        from app.services.rag.validator import ResponseValidator

        validator = ResponseValidator()

        # Test various PMID formats
        test_cases = [
            ("See [PMID: 12345]", ["12345"]),
            ("According to PMID:67890", ["67890"]),
            ("References [PMID: 11111] and [PMID: 22222]", ["11111", "22222"]),
            ("No citations here", []),
        ]

        for text, expected in test_cases:
            pmids = validator._extract_pmids(text)
            assert pmids == expected, f"Failed for: {text}"

    def test_claim_indicators(self):
        """Test claim indicators detection"""
        from app.services.rag.validator import ResponseValidator

        validator = ResponseValidator()

        # Create mock documents
        from app.services.rag.retriever import RetrievedDocument

        docs = [
            RetrievedDocument(
                chunk_id="1",
                pmid="12345",
                title="Test",
                content="Test content",
                section="abstract",
                score=0.9,
                metadata={},
            )
        ]

        # Test with claims
        response_with_claims = "The study shows that treatment works. Research indicates improvement."
        has_hallucination, suspicious = validator.check_hallucination(response_with_claims, docs)
        assert isinstance(has_hallucination, bool)


class TestI18nDeep:
    """Deep tests for i18n"""

    def test_medical_terms_mapping(self):
        """Test medical terms mapping completeness"""
        from app.core.i18n import MEDICAL_TERMS_KO_EN

        # Check key medical terms are mapped
        important_terms = ["암", "당뇨", "치료", "진단", "백신"]
        for term in important_terms:
            assert term in MEDICAL_TERMS_KO_EN, f"Missing: {term}"

    def test_translate_multiple_terms(self):
        """Test translating text with multiple terms"""
        from app.core.i18n import translate_medical_terms

        text = "폐암 치료법과 면역치료"
        result = translate_medical_terms(text, "ko_to_en")
        assert isinstance(result, str)

    def test_language_detection_edge_cases(self):
        """Test language detection edge cases"""
        from app.core.i18n import detect_language

        # Numbers only - should default to 'en'
        assert detect_language("12345") == "en"

        # Mixed with mostly Korean (more than 30% Korean)
        assert detect_language("안녕하세요 world") == "ko"

        # Special characters only - should default to 'en'
        assert detect_language("!!!") == "en"


class TestPubMedClientDeep:
    """Deep tests for PubMed client"""

    def test_parse_month_all_months(self):
        """Test parsing all month formats"""
        from app.services.pubmed.client import PubMedClient

        client = PubMedClient()

        # Test all month abbreviations
        months = ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec"]
        for i, month in enumerate(months, 1):
            assert client._parse_month(month) == i

    def test_parse_month_full_names(self):
        """Test parsing full month names"""
        from app.services.pubmed.client import PubMedClient

        client = PubMedClient()

        # Full names should also work (first 3 chars)
        assert client._parse_month("January") == 1
        assert client._parse_month("December") == 12


class TestArxivClientDeep:
    """Deep tests for arXiv client"""

    def test_arxiv_paper_optional_fields(self):
        """Test ArXivPaper with optional fields"""
        from app.services.arxiv.client import ArXivPaper

        # Minimal paper
        paper = ArXivPaper(
            arxiv_id="2401.12345",
            title="Test Paper",
        )
        assert paper.abstract is None
        assert paper.authors is None
        assert paper.pdf_url is None

    def test_to_common_format_complete(self):
        """Test complete common format conversion"""
        from app.services.arxiv.client import ArXivClient, ArXivPaper

        client = ArXivClient()
        paper = ArXivPaper(
            arxiv_id="2401.12345",
            title="Test Title",
            abstract="Test abstract",
            authors=["Author One", "Author Two"],
            published=datetime(2024, 1, 15),
            updated=datetime(2024, 1, 20),
            pdf_url="http://arxiv.org/pdf/2401.12345",
            categories=["cs.AI", "cs.LG"],
            doi="10.1234/test",
        )

        result = client.to_common_format(paper)
        assert result["id"] == "2401.12345"
        assert result["source"] == "arXiv"
        assert result["doi"] == "10.1234/test"
        assert len(result["authors"]) == 2


class TestLLMServiceDeep:
    """Deep tests for LLM service"""

    def test_llm_service_model_setting(self):
        """Test LLM service model configuration"""
        from app.services.rag.llm import LLMService

        # Default model
        service = LLMService()
        assert service.model is not None

        # Custom model
        service2 = LLMService(model="gpt-4o")
        assert service2.model == "gpt-4o"

    def test_token_counting_various_lengths(self):
        """Test token counting with various text lengths"""
        from app.services.rag.llm import LLMService

        service = LLMService()

        test_cases = [
            ("", 0),
            ("word", 1),
            ("This is a longer sentence with more words.", 10),
        ]

        for text, expected_min in test_cases:
            count = service.count_tokens(text)
            assert count >= expected_min or count == len(text) // 4


class TestPromptBuilderDeep:
    """Deep tests for prompt builder"""

    def test_format_context_with_multiple_chunks(self):
        """Test formatting context with multiple chunks"""
        from app.services.rag.prompts import format_context

        chunks = [
            {"pmid": "111", "title": "Paper 1", "section": "abstract", "content": "Content 1"},
            {"pmid": "222", "title": "Paper 2", "section": "methods", "content": "Content 2"},
            {"pmid": "333", "title": "Paper 3", "section": "results", "content": "Content 3"},
        ]

        result = format_context(chunks)
        assert "111" in result
        assert "222" in result
        assert "333" in result
        assert "---" in result  # Separator

    def test_build_prompt_different_languages(self):
        """Test building prompts in different languages"""
        from app.services.rag.prompts import build_prompt

        # English
        system_en, user_en = build_prompt("What is cancer?", "Some context", "en")
        assert "biomedical" in system_en.lower()

        # Korean
        system_ko, user_ko = build_prompt("암이란?", "컨텍스트", "ko")
        assert len(system_ko) > 0


class TestDemoServiceDeep:
    """Deep tests for demo service"""

    def test_demo_results_structure(self):
        """Test demo search results structure"""
        from app.services.demo import get_demo_search_results

        result = get_demo_search_results("test", 10)

        assert "results" in result
        assert "total" in result
        assert "query_time_ms" in result

        # Check result item structure
        if result["results"]:
            item = result["results"][0]
            assert "pmid" in item
            assert "title" in item
            assert "abstract" in item

    def test_demo_chat_structure(self):
        """Test demo chat response structure"""
        from app.services.demo import get_demo_chat_response

        result = get_demo_chat_response("test question")

        assert "answer" in result
        assert "citations" in result
        assert isinstance(result["citations"], list)
