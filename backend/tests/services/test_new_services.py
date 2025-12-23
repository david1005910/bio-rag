"""
Tests for newly integrated services from RAG_search-main
- i18n (multilingual support)
- ArXiv client
- TrendAnalyzer
- EmbeddingModelFactory
- PaperSummarizer
- Document extractor
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ==================== i18n Tests ====================
class TestI18n:
    """Multilingual support tests"""

    def test_detect_language_korean(self):
        """한국어 감지 테스트"""
        from app.core.i18n import detect_language

        assert detect_language("당뇨병 치료 연구") == 'ko'
        assert detect_language("암 면역 치료") == 'ko'
        assert detect_language("한국어 테스트") == 'ko'

    def test_detect_language_english(self):
        """영어 감지 테스트"""
        from app.core.i18n import detect_language

        assert detect_language("diabetes treatment research") == 'en'
        assert detect_language("cancer immunotherapy") == 'en'
        assert detect_language("machine learning") == 'en'

    def test_detect_language_mixed(self):
        """혼합 언어 감지 테스트"""
        from app.core.i18n import detect_language

        # 한글이 30% 이상이면 한국어로 감지
        result = detect_language("당뇨병 diabetes test")
        assert result in ['ko', 'en']  # 비율에 따라 달라짐

    def test_translate_medical_terms(self):
        """의학 용어 번역 테스트"""
        from app.core.i18n import translate_medical_terms

        result = translate_medical_terms("당뇨병 치료", 'ko_to_en')
        assert 'diabetes' in result.lower()

        result = translate_medical_terms("암 연구", 'ko_to_en')
        assert 'cancer' in result.lower()

        result = translate_medical_terms("면역치료", 'ko_to_en')
        assert 'immunotherapy' in result.lower()

    def test_multilingual_support_class(self):
        """MultilingualSupport 클래스 테스트"""
        from app.core.i18n import MultilingualSupport

        support = MultilingualSupport()
        assert support.detect_language("한국어 텍스트") == 'ko'
        assert support.detect_language("English text") == 'en'


# ==================== ArXiv Client Tests ====================
class TestArXivClient:
    """ArXiv API client tests"""

    def test_arxiv_client_init(self):
        """ArXiv 클라이언트 초기화 테스트"""
        from app.services.arxiv import ArXivClient

        client = ArXivClient(rate_limit=3)
        assert client.rate_limit == 3

    @pytest.mark.asyncio
    async def test_arxiv_search_mock(self):
        """ArXiv 검색 테스트 (mock)"""
        from app.services.arxiv import ArXivClient

        client = ArXivClient()

        # Mock response
        mock_xml = '''<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2301.00001</id>
                <title>Test Paper Title</title>
                <summary>Test abstract content</summary>
                <author><name>Test Author</name></author>
                <published>2023-01-01T00:00:00Z</published>
                <link title="pdf" href="http://arxiv.org/pdf/2301.00001"/>
                <category term="cs.AI"/>
            </entry>
        </feed>'''

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.text = mock_xml
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            papers = await client.search("machine learning", max_results=1)

            assert len(papers) == 1
            assert papers[0].title == "Test Paper Title"
            assert papers[0].arxiv_id == "2301.00001"

    def test_arxiv_to_common_format(self):
        """ArXiv 논문을 공통 포맷으로 변환 테스트"""
        from app.services.arxiv import ArXivClient, ArXivPaper
        from datetime import datetime

        client = ArXivClient()
        paper = ArXivPaper(
            arxiv_id="2301.00001",
            title="Test Paper",
            abstract="Test abstract",
            authors=["Author 1", "Author 2"],
            published=datetime(2023, 1, 1),
            pdf_url="http://arxiv.org/pdf/2301.00001"
        )

        result = client.to_common_format(paper)

        assert result['id'] == "2301.00001"
        assert result['source'] == "arXiv"
        assert result['title'] == "Test Paper"
        assert result['abstract'] == "Test abstract"
        assert result['pdf_url'] == "http://arxiv.org/pdf/2301.00001"


# ==================== TrendAnalyzer Tests ====================
class TestTrendAnalyzer:
    """Research trend analyzer tests"""

    def test_trend_analyzer_init(self):
        """TrendAnalyzer 초기화 테스트"""
        from app.services.analytics import TrendAnalyzer

        analyzer = TrendAnalyzer()
        assert analyzer.papers == []
        assert analyzer.trend_data == {}

    def test_set_papers(self):
        """논문 데이터 설정 테스트"""
        from app.services.analytics import TrendAnalyzer

        analyzer = TrendAnalyzer()
        papers = [
            {'title': 'Paper 1', 'abstract': 'Test abstract 1', 'publication_date': '2023-01-01'},
            {'title': 'Paper 2', 'abstract': 'Test abstract 2', 'publication_date': '2023-06-01'},
        ]
        analyzer.set_papers(papers)

        assert len(analyzer.papers) == 2

    def test_analyze_publication_trend(self):
        """출판 트렌드 분석 테스트"""
        from app.services.analytics import TrendAnalyzer

        analyzer = TrendAnalyzer()
        papers = [
            {'title': 'Paper 1', 'abstract': 'Test', 'publication_date': '2023-01-01'},
            {'title': 'Paper 2', 'abstract': 'Test', 'publication_date': '2023-06-01'},
            {'title': 'Paper 3', 'abstract': 'Test', 'publication_date': '2024-01-01'},
        ]
        analyzer.set_papers(papers)

        trend = analyzer.analyze_publication_trend()

        assert 'years' in trend
        assert 'counts' in trend
        assert 2023 in trend['years']
        assert 2024 in trend['years']

    def test_extract_key_terms(self):
        """핵심 키워드 추출 테스트"""
        from app.services.analytics import TrendAnalyzer

        analyzer = TrendAnalyzer()
        papers = [
            {'title': 'Test', 'abstract': 'diabetes treatment insulin glucose metabolism'},
            {'title': 'Test', 'abstract': 'diabetes prevention insulin resistance glucose'},
        ]
        analyzer.set_papers(papers)

        terms = analyzer.extract_key_terms(top_n=5)

        assert 'terms' in terms
        assert 'counts' in terms
        # 'diabetes', 'insulin', 'glucose' should be among top terms
        assert any(t in ['diabetes', 'insulin', 'glucose'] for t in terms['terms'])

    def test_generate_report(self):
        """리포트 생성 테스트"""
        from app.services.analytics import TrendAnalyzer

        analyzer = TrendAnalyzer()
        papers = [
            {'title': 'Paper 1', 'abstract': 'diabetes treatment research', 'publication_date': '2023-01-01'},
        ]
        analyzer.set_papers(papers)
        analyzer.analyze_publication_trend()
        analyzer.extract_key_terms()

        report = analyzer.generate_report("diabetes")

        assert "diabetes" in report
        assert "연구 트렌드 분석" in report


# ==================== Embedding Model Tests ====================
class TestEmbeddingModelFactory:
    """Embedding model factory tests"""

    def test_get_available_models(self):
        """사용 가능한 모델 목록 테스트"""
        from app.services.rag.embedding import EmbeddingModelFactory, EMBEDDING_MODELS

        models = EmbeddingModelFactory.get_available_models()

        assert 'pubmedbert' in models
        assert 'biobert' in models
        assert 'openai-small' in models
        assert 'minilm' in models

    def test_embedding_models_config(self):
        """임베딩 모델 설정 테스트"""
        from app.services.rag.embedding import EMBEDDING_MODELS

        # PubMedBERT
        assert EMBEDDING_MODELS['pubmedbert']['dimension'] == 768
        assert EMBEDDING_MODELS['pubmedbert']['type'] == 'huggingface'

        # OpenAI
        assert EMBEDDING_MODELS['openai-small']['dimension'] == 1536
        assert EMBEDDING_MODELS['openai-small']['type'] == 'openai'

    def test_openai_embedding_init(self):
        """OpenAI 임베딩 초기화 테스트"""
        from app.services.rag.embedding import OpenAIEmbedding

        embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")
        assert embedding.model == "text-embedding-3-small"
        assert embedding.dimension == 1536


# ==================== PaperSummarizer Tests ====================
class TestPaperSummarizer:
    """Paper summarizer tests"""

    def test_summarizer_init(self):
        """PaperSummarizer 초기화 테스트"""
        from app.services.rag.summarizer import PaperSummarizer

        summarizer = PaperSummarizer(api_key="test-key", language='ko')
        assert summarizer.language == 'ko'
        assert summarizer.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_summarize_paper_no_api_key(self):
        """API 키 없이 요약 테스트"""
        from app.services.rag.summarizer import PaperSummarizer

        summarizer = PaperSummarizer(api_key=None)
        paper = {
            'id': 'test123',
            'title': 'Test Paper',
            'abstract': 'This is a test abstract for the paper.'
        }

        result = await summarizer.summarize_paper(paper)

        assert result.paper_id == 'test123'
        assert result.success is False
        assert "API key" in (result.error or "")

    def test_system_prompts(self):
        """시스템 프롬프트 확인 테스트"""
        from app.services.rag.summarizer import PaperSummarizer

        assert 'ko' in PaperSummarizer.SYSTEM_PROMPTS
        assert 'en' in PaperSummarizer.SYSTEM_PROMPTS
        assert '한국어' in PaperSummarizer.SYSTEM_PROMPTS['ko']


# ==================== Document Extractor Tests ====================
class TestDocumentExtractor:
    """Document extraction tests"""

    def test_pdf_downloader_init(self):
        """PDFDownloader 초기화 테스트"""
        from app.services.document import PDFDownloader
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = PDFDownloader(save_dir=tmpdir)
            assert downloader.save_dir.exists()

    def test_safe_filename(self):
        """안전한 파일명 생성 테스트"""
        from app.services.document import PDFDownloader
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = PDFDownloader(save_dir=tmpdir)

            filename = downloader._safe_filename(
                "Test Paper: A Study of Something!",
                "12345"
            )

            assert "12345" in filename
            assert ":" not in filename
            assert "!" not in filename

    def test_text_extractor_txt(self):
        """TXT 파일 추출 테스트"""
        from app.services.document import TextExtractor
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for extraction")
            temp_path = f.name

        try:
            text = TextExtractor.extract_from_txt(temp_path)
            assert "Test content" in text
        finally:
            os.unlink(temp_path)

    def test_extract_all(self):
        """여러 파일 추출 테스트"""
        from app.services.document import TextExtractor
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content 1")
            temp_path1 = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content 2")
            temp_path2 = f.name

        try:
            results = TextExtractor.extract_all([temp_path1, temp_path2])
            assert len(results) == 2
            assert results[0].success is True
            assert results[1].success is True
        finally:
            os.unlink(temp_path1)
            os.unlink(temp_path2)


# ==================== Integration Tests ====================
class TestIntegration:
    """Integration tests for new services"""

    @pytest.mark.asyncio
    async def test_full_trend_analysis_flow(self):
        """전체 트렌드 분석 플로우 테스트"""
        from app.services.analytics import TrendAnalyzer

        analyzer = TrendAnalyzer()

        # 샘플 논문 데이터
        papers = [
            {
                'title': 'Diabetes Treatment Study 1',
                'abstract': 'This study examines diabetes treatment using insulin therapy.',
                'publication_date': '2023-01-15'
            },
            {
                'title': 'Diabetes Prevention Research',
                'abstract': 'Prevention strategies for type 2 diabetes including diet and exercise.',
                'publication_date': '2023-06-20'
            },
            {
                'title': 'Novel Diabetes Drug Discovery',
                'abstract': 'Discovery of new drug compounds for diabetes treatment.',
                'publication_date': '2024-02-10'
            }
        ]

        analyzer.set_papers(papers)
        result = await analyzer.analyze("diabetes")

        assert result.keyword == "diabetes"
        assert result.total_papers == 3
        assert result.year_trend is not None
        assert result.key_terms is not None

    def test_multilingual_medical_query(self):
        """다국어 의학 쿼리 테스트"""
        from app.core.i18n import detect_language, translate_medical_terms

        # 한국어 쿼리
        korean_query = "당뇨병 치료법"
        assert detect_language(korean_query) == 'ko'

        # 영어로 번역
        english_query = translate_medical_terms(korean_query, 'ko_to_en')
        assert 'diabetes' in english_query.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
