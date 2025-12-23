"""
Tests for new API endpoints
- /api/v1/arxiv
- /api/v1/analytics
- /api/v1/documents
- /api/v1/i18n
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from app.main import app


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


# ==================== i18n Endpoint Tests ====================
class TestI18nEndpoints:
    """i18n API tests"""

    def test_detect_language_korean(self, client):
        """한국어 감지 테스트"""
        response = client.get("/api/v1/i18n/detect", params={"text": "당뇨병 치료"})
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "ko"
        assert data["language_name"] == "Korean"

    def test_detect_language_english(self, client):
        """영어 감지 테스트"""
        response = client.get("/api/v1/i18n/detect", params={"text": "diabetes treatment"})
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "en"

    def test_translate_korean(self, client):
        """한국어 번역 테스트"""
        response = client.get("/api/v1/i18n/translate", params={"text": "암 면역치료"})
        assert response.status_code == 200
        data = response.json()
        assert data["source_language"] == "ko"
        assert data["target_language"] == "en"
        assert "immunotherapy" in data["translated"].lower()

    def test_translate_english_passthrough(self, client):
        """영어 패스스루 테스트"""
        response = client.get("/api/v1/i18n/translate", params={"text": "cancer treatment"})
        assert response.status_code == 200
        data = response.json()
        assert data["original"] == data["translated"]
        assert data["method"] == "none"

    def test_medical_terms(self, client):
        """의학 용어 사전 테스트"""
        response = client.get("/api/v1/i18n/medical-terms", params={"limit": 10})
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 10
        assert all("korean" in item and "english" in item for item in data)

    def test_medical_terms_search(self, client):
        """의학 용어 검색 테스트"""
        response = client.get("/api/v1/i18n/medical-terms", params={"search": "당뇨"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert any("당뇨" in item["korean"] for item in data)

    def test_supported_languages(self, client):
        """지원 언어 조회 테스트"""
        response = client.get("/api/v1/i18n/supported-languages")
        assert response.status_code == 200
        data = response.json()
        assert "languages" in data
        assert len(data["languages"]) == 2

    def test_translate_query(self, client):
        """검색 쿼리 번역 테스트"""
        response = client.post(
            "/api/v1/i18n/translate-query",
            json={"query": "당뇨병 예방"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "ko"
        assert data["is_translated"] is True


# ==================== ArXiv Endpoint Tests ====================
class TestArXivEndpoints:
    """ArXiv API tests"""

    def test_arxiv_search_get(self, client):
        """ArXiv 검색 GET 테스트"""
        with patch("app.services.arxiv.arxiv_client.search") as mock_search:
            mock_search.return_value = []
            response = client.get(
                "/api/v1/arxiv/search",
                params={"query": "machine learning", "max_results": 5}
            )
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert data["query"] == "machine learning"

    def test_arxiv_search_post(self, client):
        """ArXiv 검색 POST 테스트"""
        with patch("app.services.arxiv.arxiv_client.search") as mock_search:
            mock_search.return_value = []
            response = client.post(
                "/api/v1/arxiv/search",
                json={"query": "diabetes", "max_results": 3}
            )
            assert response.status_code == 200

    def test_arxiv_paper_not_found(self, client):
        """ArXiv 논문 없음 테스트"""
        with patch("app.services.arxiv.arxiv_client.fetch_paper") as mock_fetch:
            mock_fetch.return_value = None
            response = client.get("/api/v1/arxiv/paper/9999.99999")
            assert response.status_code == 404


# ==================== Analytics Endpoint Tests ====================
class TestAnalyticsEndpoints:
    """Analytics API tests"""

    def test_trends_from_papers(self, client):
        """제공된 논문으로 트렌드 분석 테스트"""
        papers = [
            {
                "title": "Diabetes Study",
                "abstract": "Treatment of diabetes using insulin therapy.",
                "publication_date": "2023-01-15"
            },
            {
                "title": "Cancer Research",
                "abstract": "Novel immunotherapy for cancer.",
                "publication_date": "2023-06-20"
            }
        ]
        response = client.post(
            "/api/v1/analytics/trends/from-papers",
            params={"keyword": "medical"},
            json=papers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["keyword"] == "medical"
        assert data["total_papers"] == 2

    def test_trends_empty_papers(self, client):
        """빈 논문 리스트 테스트"""
        response = client.post(
            "/api/v1/analytics/trends/from-papers",
            params={"keyword": "test"},
            json=[]
        )
        assert response.status_code == 400


# ==================== Documents Endpoint Tests ====================
class TestDocumentsEndpoints:
    """Documents API tests"""

    def test_download_no_papers(self, client):
        """빈 논문 리스트 다운로드 테스트"""
        response = client.post(
            "/api/v1/documents/download",
            json={"papers": []}
        )
        assert response.status_code == 400

    def test_download_too_many(self, client):
        """너무 많은 논문 다운로드 테스트"""
        papers = [{"id": str(i), "title": f"Paper {i}"} for i in range(60)]
        response = client.post(
            "/api/v1/documents/download",
            json={"papers": papers}
        )
        assert response.status_code == 400

    def test_extract_no_files(self, client):
        """빈 파일 리스트 추출 테스트"""
        response = client.post(
            "/api/v1/documents/extract",
            json={"filepaths": []}
        )
        assert response.status_code == 400

    def test_extract_nonexistent_files(self, client):
        """존재하지 않는 파일 추출 테스트"""
        response = client.post(
            "/api/v1/documents/extract",
            json={"filepaths": ["/nonexistent/file.pdf"]}
        )
        assert response.status_code == 404

    def test_summarize(self, client):
        """논문 요약 테스트"""
        response = client.post(
            "/api/v1/documents/summarize",
            json={
                "paper": {
                    "id": "test123",
                    "title": "Test Paper",
                    "abstract": "This is a test abstract."
                },
                "language": "en"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["paper_id"] == "test123"

    def test_summarize_batch_empty(self, client):
        """빈 배치 요약 테스트"""
        response = client.post(
            "/api/v1/documents/summarize-batch",
            params={"language": "en"},
            json=[]
        )
        assert response.status_code == 400

    def test_summarize_batch_too_many(self, client):
        """너무 많은 배치 요약 테스트"""
        papers = [{"id": str(i), "title": f"Paper {i}", "abstract": "Test"} for i in range(15)]
        response = client.post(
            "/api/v1/documents/summarize-batch",
            params={"language": "en"},
            json=papers
        )
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
