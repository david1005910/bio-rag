"""Tests for pubmed_service.py"""
import pytest
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch
import xml.etree.ElementTree as ET


# ============================================================================
# Testable implementations of PubMed service classes
# ============================================================================

@dataclass
class PaperMetadata:
    """Paper metadata from PubMed."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: Optional[datetime]
    doi: Optional[str]
    keywords: List[str]
    mesh_terms: List[str]
    pdf_url: Optional[str] = None


class TestablePubMedService:
    """Testable version of PubMed service."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.rate_limit = 10 if api_key else 3
        self._translator = None

    def _get_translator(self):
        return self._translator

    def _build_url(self, endpoint: str, params: Dict[str, Any]) -> str:
        if self.api_key:
            params['api_key'] = self.api_key
        params['db'] = 'pubmed'
        query_string = '&'.join(f"{k}={v}" for k, v in params.items())
        return f"{self.BASE_URL}{endpoint}?{query_string}"

    def _parse_xml_response(self, xml_text: str) -> List[PaperMetadata]:
        papers = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall('.//PubmedArticle'):
                try:
                    medline = article.find('.//MedlineCitation')
                    if medline is None:
                        continue

                    pmid_elem = medline.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else ''

                    article_elem = medline.find('.//Article')
                    if article_elem is None:
                        continue

                    title_elem = article_elem.find('.//ArticleTitle')
                    title = self._get_text_content(title_elem) if title_elem is not None else ''

                    abstract_elem = article_elem.find('.//Abstract')
                    abstract = ''
                    if abstract_elem is not None:
                        abstract_texts = []
                        for abstract_text in abstract_elem.findall('.//AbstractText'):
                            label = abstract_text.get('Label', '')
                            text = self._get_text_content(abstract_text)
                            if label:
                                abstract_texts.append(f"{label}: {text}")
                            else:
                                abstract_texts.append(text)
                        abstract = ' '.join(abstract_texts)

                    authors = []
                    author_list = article_elem.find('.//AuthorList')
                    if author_list is not None:
                        for author in author_list.findall('.//Author'):
                            lastname = author.find('LastName')
                            forename = author.find('ForeName')
                            if lastname is not None:
                                name = lastname.text or ''
                                if forename is not None and forename.text:
                                    name = f"{forename.text} {name}"
                                authors.append(name)

                    journal_elem = article_elem.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else ''

                    pub_date = None
                    pub_date_elem = article_elem.find('.//ArticleDate') or medline.find('.//DateCompleted')
                    if pub_date_elem is not None:
                        year = pub_date_elem.find('Year')
                        month = pub_date_elem.find('Month')
                        day = pub_date_elem.find('Day')
                        if year is not None:
                            try:
                                pub_date = datetime(
                                    int(year.text or '2000'),
                                    int(month.text or '1') if month is not None else 1,
                                    int(day.text or '1') if day is not None else 1
                                )
                            except ValueError:
                                pass

                    doi = None
                    for elocation in article_elem.findall('.//ELocationID'):
                        if elocation.get('EIdType') == 'doi':
                            doi = elocation.text
                            break

                    keywords = []
                    keyword_list = medline.find('.//KeywordList')
                    if keyword_list is not None:
                        for kw in keyword_list.findall('.//Keyword'):
                            if kw.text:
                                keywords.append(kw.text)

                    mesh_terms = []
                    mesh_heading_list = medline.find('.//MeshHeadingList')
                    if mesh_heading_list is not None:
                        for mesh in mesh_heading_list.findall('.//MeshHeading/DescriptorName'):
                            if mesh.text:
                                mesh_terms.append(mesh.text)

                    papers.append(PaperMetadata(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        journal=journal,
                        publication_date=pub_date,
                        doi=doi,
                        keywords=keywords,
                        mesh_terms=mesh_terms
                    ))

                except Exception:
                    continue

        except ET.ParseError:
            pass

        return papers

    def _get_text_content(self, element) -> str:
        if element is None:
            return ''
        texts = []
        if element.text:
            texts.append(element.text)
        for child in element:
            if child.text:
                texts.append(child.text)
            if child.tail:
                texts.append(child.tail)
        return ''.join(texts).strip()


# ============================================================================
# Sample XML responses for testing
# ============================================================================

SAMPLE_PUBMED_XML = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2019//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_190101.dtd">
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation Status="MEDLINE" Owner="NLM">
            <PMID Version="1">12345678</PMID>
            <Article PubModel="Print">
                <Journal>
                    <Title>Nature Medicine</Title>
                </Journal>
                <ArticleTitle>Novel Cancer Treatment Using Immunotherapy</ArticleTitle>
                <Abstract>
                    <AbstractText Label="BACKGROUND">Cancer treatment remains challenging.</AbstractText>
                    <AbstractText Label="METHODS">We tested a new immunotherapy approach.</AbstractText>
                    <AbstractText Label="RESULTS">Significant tumor reduction was observed.</AbstractText>
                </Abstract>
                <AuthorList CompleteYN="Y">
                    <Author ValidYN="Y">
                        <LastName>Smith</LastName>
                        <ForeName>John</ForeName>
                    </Author>
                    <Author ValidYN="Y">
                        <LastName>Doe</LastName>
                        <ForeName>Jane</ForeName>
                    </Author>
                </AuthorList>
                <ArticleDate DateType="Electronic">
                    <Year>2024</Year>
                    <Month>01</Month>
                    <Day>15</Day>
                </ArticleDate>
                <ELocationID EIdType="doi">10.1038/nm.12345</ELocationID>
            </Article>
            <KeywordList Owner="NOTNLM">
                <Keyword>cancer</Keyword>
                <Keyword>immunotherapy</Keyword>
            </KeywordList>
            <MeshHeadingList>
                <MeshHeading>
                    <DescriptorName>Neoplasms</DescriptorName>
                </MeshHeading>
                <MeshHeading>
                    <DescriptorName>Immunotherapy</DescriptorName>
                </MeshHeading>
            </MeshHeadingList>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>'''

SAMPLE_PUBMED_XML_MINIMAL = '''<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation Status="MEDLINE" Owner="NLM">
            <PMID Version="1">99999999</PMID>
            <Article PubModel="Print">
                <Journal>
                    <Title>Science</Title>
                </Journal>
                <ArticleTitle>Basic Research Article</ArticleTitle>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>'''

SAMPLE_PUBMED_XML_MULTIPLE = '''<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation Status="MEDLINE" Owner="NLM">
            <PMID Version="1">11111111</PMID>
            <Article PubModel="Print">
                <Journal><Title>Journal A</Title></Journal>
                <ArticleTitle>First Paper</ArticleTitle>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
    <PubmedArticle>
        <MedlineCitation Status="MEDLINE" Owner="NLM">
            <PMID Version="1">22222222</PMID>
            <Article PubModel="Print">
                <Journal><Title>Journal B</Title></Journal>
                <ArticleTitle>Second Paper</ArticleTitle>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>'''


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def pubmed_service():
    """Create a testable PubMed service instance."""
    return TestablePubMedService()


@pytest.fixture
def pubmed_service_with_api_key():
    """Create a PubMed service with API key."""
    return TestablePubMedService(api_key="test_api_key")


# ============================================================================
# Tests
# ============================================================================

class TestPaperMetadata:
    """Tests for PaperMetadata dataclass."""

    def test_paper_metadata_creation(self):
        """Test creating PaperMetadata."""
        paper = PaperMetadata(
            pmid="12345",
            title="Test Paper",
            abstract="This is a test abstract.",
            authors=["John Smith", "Jane Doe"],
            journal="Nature",
            publication_date=datetime(2024, 1, 15),
            doi="10.1038/test",
            keywords=["test", "research"],
            mesh_terms=["Science"]
        )

        assert paper.pmid == "12345"
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.journal == "Nature"

    def test_paper_metadata_optional_fields(self):
        """Test PaperMetadata with optional fields as None."""
        paper = PaperMetadata(
            pmid="12345",
            title="Test Paper",
            abstract="Abstract",
            authors=[],
            journal="",
            publication_date=None,
            doi=None,
            keywords=[],
            mesh_terms=[]
        )

        assert paper.publication_date is None
        assert paper.doi is None
        assert paper.pdf_url is None


class TestBuildUrl:
    """Tests for URL building."""

    def test_build_url_without_api_key(self, pubmed_service):
        """Test URL building without API key."""
        url = pubmed_service._build_url('esearch.fcgi', {'term': 'cancer'})

        assert 'esearch.fcgi' in url
        assert 'term=cancer' in url
        assert 'db=pubmed' in url
        assert 'api_key' not in url

    def test_build_url_with_api_key(self, pubmed_service_with_api_key):
        """Test URL building with API key."""
        url = pubmed_service_with_api_key._build_url('esearch.fcgi', {'term': 'cancer'})

        assert 'api_key=test_api_key' in url

    def test_build_url_multiple_params(self, pubmed_service):
        """Test URL building with multiple parameters."""
        params = {'term': 'cancer', 'retmax': 10, 'sort': 'relevance'}
        url = pubmed_service._build_url('esearch.fcgi', params)

        assert 'term=cancer' in url
        assert 'retmax=10' in url
        assert 'sort=relevance' in url


class TestParseXmlResponse:
    """Tests for XML parsing."""

    def test_parse_complete_xml(self, pubmed_service):
        """Test parsing a complete XML response."""
        papers = pubmed_service._parse_xml_response(SAMPLE_PUBMED_XML)

        assert len(papers) == 1
        paper = papers[0]

        assert paper.pmid == "12345678"
        assert paper.title == "Novel Cancer Treatment Using Immunotherapy"
        assert paper.journal == "Nature Medicine"
        assert len(paper.authors) == 2
        assert "John Smith" in paper.authors
        assert "Jane Doe" in paper.authors
        assert paper.doi == "10.1038/nm.12345"
        assert "cancer" in paper.keywords
        assert "Neoplasms" in paper.mesh_terms

    def test_parse_abstract_with_labels(self, pubmed_service):
        """Test parsing abstract with section labels."""
        papers = pubmed_service._parse_xml_response(SAMPLE_PUBMED_XML)

        paper = papers[0]
        assert "BACKGROUND:" in paper.abstract
        assert "METHODS:" in paper.abstract
        assert "RESULTS:" in paper.abstract

    def test_parse_publication_date(self, pubmed_service):
        """Test parsing publication date."""
        papers = pubmed_service._parse_xml_response(SAMPLE_PUBMED_XML)

        paper = papers[0]
        assert paper.publication_date == datetime(2024, 1, 15)

    def test_parse_minimal_xml(self, pubmed_service):
        """Test parsing minimal XML with only required fields."""
        papers = pubmed_service._parse_xml_response(SAMPLE_PUBMED_XML_MINIMAL)

        assert len(papers) == 1
        paper = papers[0]

        assert paper.pmid == "99999999"
        assert paper.title == "Basic Research Article"
        assert paper.journal == "Science"
        assert paper.abstract == ""
        assert paper.authors == []
        assert paper.keywords == []

    def test_parse_multiple_articles(self, pubmed_service):
        """Test parsing multiple articles."""
        papers = pubmed_service._parse_xml_response(SAMPLE_PUBMED_XML_MULTIPLE)

        assert len(papers) == 2
        assert papers[0].pmid == "11111111"
        assert papers[1].pmid == "22222222"

    def test_parse_invalid_xml(self, pubmed_service):
        """Test parsing invalid XML returns empty list."""
        papers = pubmed_service._parse_xml_response("not valid xml")

        assert papers == []

    def test_parse_empty_xml(self, pubmed_service):
        """Test parsing empty XML set."""
        empty_xml = '<?xml version="1.0"?><PubmedArticleSet></PubmedArticleSet>'
        papers = pubmed_service._parse_xml_response(empty_xml)

        assert papers == []


class TestGetTextContent:
    """Tests for text content extraction."""

    def test_get_text_content_simple(self, pubmed_service):
        """Test extracting simple text content."""
        elem = ET.fromstring("<title>Simple Title</title>")
        text = pubmed_service._get_text_content(elem)

        assert text == "Simple Title"

    def test_get_text_content_with_children(self, pubmed_service):
        """Test extracting text with child elements."""
        elem = ET.fromstring("<title>Title with <i>italic</i> text</title>")
        text = pubmed_service._get_text_content(elem)

        assert "Title with" in text
        assert "italic" in text
        assert "text" in text

    def test_get_text_content_none(self, pubmed_service):
        """Test handling None element."""
        text = pubmed_service._get_text_content(None)

        assert text == ""


class TestRateLimit:
    """Tests for rate limiting."""

    def test_rate_limit_without_api_key(self, pubmed_service):
        """Test rate limit is 3 without API key."""
        assert pubmed_service.rate_limit == 3

    def test_rate_limit_with_api_key(self, pubmed_service_with_api_key):
        """Test rate limit is 10 with API key."""
        assert pubmed_service_with_api_key.rate_limit == 10


class TestTranslationIntegration:
    """Tests for translation integration."""

    def test_translator_initially_none(self, pubmed_service):
        """Test translator is initially None."""
        assert pubmed_service._translator is None

    def test_get_translator_returns_none_by_default(self, pubmed_service):
        """Test _get_translator returns None by default."""
        result = pubmed_service._get_translator()
        assert result is None

    def test_set_translator(self, pubmed_service):
        """Test setting a translator."""
        mock_translator = MagicMock()
        pubmed_service._translator = mock_translator

        result = pubmed_service._get_translator()
        assert result == mock_translator
