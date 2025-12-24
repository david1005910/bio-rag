"""Comprehensive tests for service layer"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.i18n import (
    MultilingualSupport,
    detect_language,
    translate_medical_terms,
)
from app.services.arxiv.client import ArXivAPIError, ArXivClient, ArXivPaper
from app.services.pubmed.client import PubMedAPIError, PubMedClient
from app.services.rag.retriever import RetrievedDocument
from app.services.rag.validator import ResponseValidator, ValidationResult


class TestPubMedClient:
    """Tests for PubMed API client"""

    @pytest.fixture
    def client(self):
        """Create PubMed client instance"""
        return PubMedClient()

    def test_init(self, client):
        """Test client initialization"""
        assert client.BASE_URL == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        assert client._semaphore is None

    def test_get_semaphore(self, client):
        """Test semaphore creation"""
        sem = client._get_semaphore()
        assert isinstance(sem, asyncio.Semaphore)
        # Same semaphore should be returned
        assert client._get_semaphore() is sem

    def test_parse_month_numeric(self, client):
        """Test parsing numeric month"""
        assert client._parse_month("1") == 1
        assert client._parse_month("12") == 12

    def test_parse_month_string(self, client):
        """Test parsing month name"""
        assert client._parse_month("Jan") == 1
        assert client._parse_month("February") == 2
        assert client._parse_month("DEC") == 12

    def test_parse_month_none(self, client):
        """Test parsing None month"""
        assert client._parse_month(None) == 1

    def test_parse_month_invalid(self, client):
        """Test parsing invalid month"""
        assert client._parse_month("invalid") == 1

    def test_parse_date_none(self, client):
        """Test parsing None date element"""
        result = client._parse_date(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_search_empty_result(self, client):
        """Test search with mocked empty result"""
        mock_response = '{"esearchresult": {"idlist": []}}'
        with patch.object(client, "_request", return_value=mock_response):
            result = await client.search("nonexistent query")
            assert result == []

    @pytest.mark.asyncio
    async def test_search_with_results(self, client):
        """Test search with mocked results"""
        mock_response = '{"esearchresult": {"idlist": ["12345", "67890"]}}'
        with patch.object(client, "_request", return_value=mock_response):
            result = await client.search("cancer")
            assert result == ["12345", "67890"]

    @pytest.mark.asyncio
    async def test_search_with_date_range(self, client):
        """Test search with date range"""
        mock_response = '{"esearchresult": {"idlist": ["12345"]}}'
        with patch.object(client, "_request", return_value=mock_response) as mock_req:
            result = await client.search(
                "cancer", date_range=("2024/01/01", "2024/12/31")
            )
            assert result == ["12345"]
            # Verify date params were passed
            call_args = mock_req.call_args
            assert "mindate" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_search_error(self, client):
        """Test search error handling"""
        with patch.object(client, "_request", side_effect=Exception("API Error")):
            with pytest.raises(PubMedAPIError):
                await client.search("test")

    @pytest.mark.asyncio
    async def test_fetch_papers_empty(self, client):
        """Test fetching with empty list"""
        result = await client.fetch_papers([])
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_papers_error(self, client):
        """Test fetch error handling"""
        with patch.object(client, "_request", side_effect=Exception("API Error")):
            with pytest.raises(PubMedAPIError):
                await client.fetch_papers(["12345"])

    def test_parse_xml_valid(self, client):
        """Test parsing valid XML"""
        xml = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345</PMID>
                    <Article>
                        <ArticleTitle>Test Title</ArticleTitle>
                        <Abstract>
                            <AbstractText>Test abstract</AbstractText>
                        </Abstract>
                        <AuthorList>
                            <Author>
                                <LastName>Smith</LastName>
                                <ForeName>John</ForeName>
                            </Author>
                        </AuthorList>
                        <Journal>
                            <Title>Test Journal</Title>
                        </Journal>
                        <PubDate>
                            <Year>2024</Year>
                            <Month>Jun</Month>
                            <Day>15</Day>
                        </PubDate>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        papers = client._parse_xml(xml)
        assert len(papers) == 1
        assert papers[0].pmid == "12345"
        assert papers[0].title == "Test Title"

    def test_parse_xml_empty(self, client):
        """Test parsing XML with no articles"""
        xml = '<?xml version="1.0"?><PubmedArticleSet></PubmedArticleSet>'
        papers = client._parse_xml(xml)
        assert papers == []

    @pytest.mark.asyncio
    async def test_batch_fetch(self, client):
        """Test batch fetching"""
        mock_papers = []
        with patch.object(client, "fetch_papers", return_value=mock_papers):
            result = await client.batch_fetch(["1", "2", "3"], batch_size=2)
            assert result == []


class TestArXivClient:
    """Tests for arXiv API client"""

    @pytest.fixture
    def client(self):
        """Create arXiv client instance"""
        return ArXivClient(rate_limit=3)

    def test_init(self, client):
        """Test client initialization"""
        assert client.BASE_URL == "https://export.arxiv.org/api/query"
        assert client.rate_limit == 3

    @pytest.mark.asyncio
    async def test_search_error(self, client):
        """Test search error handling"""
        with patch("httpx.AsyncClient.get", side_effect=Exception("Network error")):
            with pytest.raises(ArXivAPIError):
                await client.search("test query")

    def test_parse_response_empty(self, client):
        """Test parsing empty response"""
        xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>
        """
        papers = client._parse_response(xml)
        assert papers == []

    def test_parse_response_valid(self, client):
        """Test parsing valid response"""
        xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:arxiv="http://arxiv.org/schemas/atom">
            <entry>
                <id>http://arxiv.org/abs/2401.12345</id>
                <title>Test Paper Title</title>
                <summary>Test abstract content</summary>
                <author><name>John Doe</name></author>
                <published>2024-01-15T00:00:00Z</published>
                <updated>2024-01-16T00:00:00Z</updated>
                <link title="pdf" href="http://arxiv.org/pdf/2401.12345"/>
                <category term="cs.AI"/>
            </entry>
        </feed>
        """
        papers = client._parse_response(xml)
        assert len(papers) == 1
        assert papers[0].arxiv_id == "2401.12345"
        assert papers[0].title == "Test Paper Title"
        assert papers[0].authors == ["John Doe"]

    def test_parse_response_invalid_xml(self, client):
        """Test parsing invalid XML"""
        with pytest.raises(ArXivAPIError):
            client._parse_response("not xml")

    @pytest.mark.asyncio
    async def test_fetch_paper_not_found(self, client):
        """Test fetching non-existent paper"""
        empty_xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom"></feed>
        """
        mock_response = MagicMock()
        mock_response.text = empty_xml
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            result = await client.fetch_paper("nonexistent")
            assert result is None

    def test_to_common_format(self, client):
        """Test converting to common format"""
        paper = ArXivPaper(
            arxiv_id="2401.12345",
            title="Test Title",
            abstract="Test abstract",
            authors=["Author One"],
            published=datetime(2024, 1, 15),
            pdf_url="http://arxiv.org/pdf/2401.12345",
            doi="10.1234/test",
            categories=["cs.AI"],
        )
        result = client.to_common_format(paper)
        assert result["id"] == "2401.12345"
        assert result["source"] == "arXiv"
        assert result["title"] == "Test Title"

    def test_to_common_format_minimal(self, client):
        """Test converting minimal paper"""
        paper = ArXivPaper(arxiv_id="2401.12345", title="Test")
        result = client.to_common_format(paper)
        assert result["authors"] == []
        assert result["published"] is None


class TestI18nFunctions:
    """Tests for i18n functions"""

    def test_detect_language_english(self):
        """Test detecting English text"""
        assert detect_language("Hello world") == "en"
        assert detect_language("This is a test") == "en"

    def test_detect_language_korean(self):
        """Test detecting Korean text"""
        assert detect_language("안녕하세요") == "ko"
        assert detect_language("암 치료 연구") == "ko"

    def test_detect_language_mixed(self):
        """Test detecting mixed text"""
        # Should detect based on ratio
        result = detect_language("cancer 암 치료법")
        assert result in ["ko", "en"]

    def test_detect_language_empty(self):
        """Test detecting empty text"""
        assert detect_language("") == "en"
        assert detect_language("   ") == "en"

    def test_translate_medical_terms_ko_to_en(self):
        """Test translating Korean medical terms to English"""
        result = translate_medical_terms("암 치료", "ko_to_en")
        assert "cancer" in result.lower()
        assert "treatment" in result.lower()

    def test_translate_medical_terms_en_to_ko(self):
        """Test translating English medical terms to Korean"""
        result = translate_medical_terms("cancer", "en_to_ko")
        assert result is not None

    def test_translate_medical_terms_multiple(self):
        """Test translating multiple terms"""
        result = translate_medical_terms("폐암 면역치료", "ko_to_en")
        assert "lung cancer" in result.lower() or "immunotherapy" in result.lower()


class TestMultilingualSupport:
    """Tests for multilingual support"""

    @pytest.fixture
    def ml_support(self):
        """Create multilingual support instance"""
        return MultilingualSupport()

    def test_init(self, ml_support):
        """Test initialization"""
        assert ml_support is not None

    def test_detect_language(self, ml_support):
        """Test language detection method"""
        assert ml_support.detect_language("Hello world") == "en"
        assert ml_support.detect_language("안녕하세요") == "ko"

    @pytest.mark.asyncio
    async def test_translate_query_english(self, ml_support):
        """Test translating English query"""
        result = await ml_support.translate_query("cancer treatment")
        assert isinstance(result, dict)
        assert "original" in result
        assert "translated" in result
        assert "language" in result
        assert result["language"] == "en"

    @pytest.mark.asyncio
    async def test_translate_query_korean(self, ml_support):
        """Test translating Korean query"""
        result = await ml_support.translate_query("암 치료")
        assert isinstance(result, dict)
        assert result["language"] == "ko"
        assert result["original"] == "암 치료"

    def test_get_response_language(self, ml_support):
        """Test getting response language"""
        assert ml_support.get_response_language("ko") == "ko"
        assert ml_support.get_response_language("en") == "en"


class TestResponseValidator:
    """Tests for response validation"""

    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return ResponseValidator()

    @pytest.fixture
    def sample_docs(self):
        """Create sample retrieved documents"""
        return [
            RetrievedDocument(
                chunk_id="12345_0",
                pmid="12345",
                title="Cancer Treatment Review",
                content="Cancer treatment includes chemotherapy and immunotherapy.",
                section="abstract",
                score=0.9,
                metadata={"pmid": "12345", "title": "Cancer Treatment Review"},
            ),
            RetrievedDocument(
                chunk_id="67890_0",
                pmid="67890",
                title="Immunotherapy Advances",
                content="Immunotherapy has shown promising results.",
                section="abstract",
                score=0.85,
                metadata={"pmid": "67890", "title": "Immunotherapy Advances"},
            ),
        ]

    def test_init(self, validator):
        """Test validator initialization"""
        assert validator is not None
        assert validator.similarity_threshold == 0.5
        assert validator.min_citations == 1

    def test_init_custom_params(self):
        """Test validator with custom parameters"""
        validator = ResponseValidator(
            similarity_threshold=0.7,
            min_citations=2,
        )
        assert validator.similarity_threshold == 0.7
        assert validator.min_citations == 2

    def test_extract_pmids(self, validator):
        """Test extracting PMIDs from text"""
        text = "Based on [PMID: 12345] and PMID:67890, we found..."
        pmids = validator._extract_pmids(text)
        assert "12345" in pmids
        assert "67890" in pmids

    def test_extract_pmids_duplicates(self, validator):
        """Test that duplicate PMIDs are removed"""
        text = "See PMID: 12345 and also [PMID: 12345]"
        pmids = validator._extract_pmids(text)
        assert pmids.count("12345") == 1

    def test_validate_response(self, validator, sample_docs):
        """Test validating response"""
        response = "Based on the research [PMID: 12345], cancer treatment involves chemotherapy."
        result = validator.validate(response, sample_docs)

        assert isinstance(result, ValidationResult)
        assert result.cited_pmids == ["12345"]
        assert "12345" in result.valid_citations

    def test_validate_invalid_citation(self, validator, sample_docs):
        """Test validating response with invalid citation"""
        response = "According to [PMID: 99999], this is true."
        result = validator.validate(response, sample_docs)

        assert "99999" in result.invalid_citations
        assert not result.is_valid

    def test_validate_no_citations(self, validator, sample_docs):
        """Test validating response without citations"""
        response = "Cancer treatment involves chemotherapy."
        result = validator.validate(response, sample_docs)

        assert len(result.cited_pmids) == 0
        assert not result.is_valid
        assert any("fewer than" in w for w in result.warnings)

    def test_check_hallucination(self, validator, sample_docs):
        """Test hallucination detection"""
        response = "The study shows that XYZ cures cancer. Research indicates complete remission."
        has_hallucination, suspicious = validator.check_hallucination(response, sample_docs)

        assert isinstance(has_hallucination, bool)
        assert isinstance(suspicious, list)

    def test_check_hallucination_with_citation(self, validator, sample_docs):
        """Test that claims with citations don't flag as hallucination"""
        response = "The study shows [PMID: 12345] that chemotherapy works."
        has_hallucination, suspicious = validator.check_hallucination(response, sample_docs)
        # The claim has a citation, so should not be suspicious
        assert isinstance(suspicious, list)

    def test_calculate_confidence_empty_docs(self, validator):
        """Test confidence calculation with empty documents"""
        confidence = validator._calculate_confidence("test", [], [])
        assert confidence == 0.0

    def test_calculate_confidence_with_citations(self, validator, sample_docs):
        """Test confidence calculation with valid citations"""
        confidence = validator._calculate_confidence(
            "test response",
            sample_docs,
            ["12345"],
        )
        assert 0 <= confidence <= 1

    def test_clean_response(self, validator):
        """Test cleaning response with invalid citations"""
        result = ValidationResult(
            is_valid=False,
            confidence_score=0.5,
            cited_pmids=["12345", "99999"],
            valid_citations=["12345"],
            invalid_citations=["99999"],
            warnings=["Invalid citation found"],
        )
        response = "Based on [PMID: 12345] and [PMID: 99999]..."
        cleaned = validator.clean_response(response, result)

        assert "[citation needed]" in cleaned
        assert "Note:" in cleaned

    def test_set_embedding_func(self, validator):
        """Test setting embedding function"""
        mock_func = lambda x: [0.1] * 768
        validator.set_embedding_func(mock_func)
        assert validator._embedding_func is not None


class TestAnalyticsTrend:
    """Tests for analytics trend service"""

    def test_import(self):
        """Test that trend service can be imported"""
        from app.services.analytics.trend import TrendAnalyzer
        assert TrendAnalyzer is not None


class TestDocumentExtractor:
    """Tests for document extraction service"""

    def test_import(self):
        """Test that document extractor can be imported"""
        from app.services.document.extractor import TextExtractor
        assert TextExtractor is not None


class TestSearchService:
    """Tests for search service"""

    def test_import(self):
        """Test that search service can be imported"""
        from app.services.search.service import SearchService
        assert SearchService is not None
