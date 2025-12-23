"""Analytics API endpoints - Research Trend Analysis"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.i18n import detect_language, translate_medical_terms
from app.services.analytics import TrendAnalyzer, TrendData
from app.services.arxiv import arxiv_client
from app.services.pubmed.client import pubmed_client

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# ==================== Request/Response Models ====================
class TrendAnalysisRequest(BaseModel):
    """트렌드 분석 요청"""
    keyword: str = Field(..., min_length=1, max_length=200, description="분석 키워드")
    max_papers: int = Field(default=50, ge=5, le=200, description="분석할 최대 논문 수")
    source: str = Field(default="pubmed", description="검색 소스 (pubmed, arxiv, both)")
    include_ai_summary: bool = Field(default=True, description="AI 요약 포함 여부")


class TrendAnalysisResponse(BaseModel):
    """트렌드 분석 결과"""
    keyword: str
    translated_keyword: str | None = None
    original_language: str
    total_papers: int
    year_trend: dict[str, Any] | None = None
    key_terms: dict[str, Any] | None = None
    emerging_topics: list[dict[str, Any]] | None = None
    content_summary: dict[str, Any] | None = None
    report: str | None = None


class QuickTrendResponse(BaseModel):
    """빠른 트렌드 분석 결과"""
    keyword: str
    total_papers: int
    years: list[int]
    counts: list[int]
    top_terms: list[str]


# ==================== Endpoints ====================
@router.post("/trends", response_model=TrendAnalysisResponse)
async def analyze_trends(request: TrendAnalysisRequest) -> TrendAnalysisResponse:
    """
    연구 트렌드 분석

    - **keyword**: 분석할 키워드 (한글 가능)
    - **max_papers**: 분석할 최대 논문 수 (5-200)
    - **source**: 검색 소스 (pubmed, arxiv, both)
    - **include_ai_summary**: AI 요약 포함 여부

    키워드 기반으로 연구 동향을 분석하고:
    - 연도별 출판 트렌드
    - 핵심 키워드 추출
    - 급부상 주제 식별
    - AI 기반 연구 내용 요약 (OpenAI API 필요)
    """
    # 언어 감지 및 번역
    original_language = detect_language(request.keyword)
    search_keyword = request.keyword

    if original_language == 'ko':
        search_keyword = translate_medical_terms(request.keyword, 'ko_to_en')

    # 논문 수집
    papers: list[dict[str, Any]] = []

    try:
        if request.source in ['pubmed', 'both']:
            # PubMed 검색
            pmids = await pubmed_client.search(search_keyword, max_results=request.max_papers)
            if pmids:
                pubmed_papers = await pubmed_client.fetch_papers(pmids[:request.max_papers])
                for p in pubmed_papers:
                    papers.append({
                        'id': p.pmid,
                        'title': p.title,
                        'abstract': p.abstract,
                        'authors': p.authors or [],
                        'publication_date': p.publication_date,
                        'source': 'PubMed',
                    })

        if request.source in ['arxiv', 'both']:
            # ArXiv 검색
            arxiv_papers = await arxiv_client.search(
                search_keyword,
                max_results=request.max_papers
            )
            for p in arxiv_papers:
                papers.append({
                    'id': p.arxiv_id,
                    'title': p.title,
                    'abstract': p.abstract,
                    'authors': p.authors or [],
                    'publication_date': p.published,
                    'source': 'arXiv',
                })

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch papers: {str(e)}"
        )

    if not papers:
        raise HTTPException(
            status_code=404,
            detail=f"No papers found for keyword: {request.keyword}"
        )

    # 트렌드 분석
    openai_key = settings.OPENAI_API_KEY if request.include_ai_summary else None
    if openai_key and openai_key.startswith("your-"):
        openai_key = None

    analyzer = TrendAnalyzer(openai_api_key=openai_key)
    analyzer.set_papers(papers)

    try:
        result = await analyzer.analyze(request.keyword)
        report = analyzer.generate_report(request.keyword)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    return TrendAnalysisResponse(
        keyword=request.keyword,
        translated_keyword=search_keyword if original_language == 'ko' else None,
        original_language=original_language,
        total_papers=result.total_papers,
        year_trend=result.year_trend,
        key_terms=result.key_terms,
        emerging_topics=result.emerging_topics,
        content_summary=result.content_summary,
        report=report,
    )


@router.get("/trends/quick", response_model=QuickTrendResponse)
async def quick_trend_analysis(
    keyword: str = Query(..., min_length=1, max_length=200, description="분석 키워드"),
    max_papers: int = Query(default=30, ge=5, le=100, description="논문 수"),
) -> QuickTrendResponse:
    """
    빠른 트렌드 분석 (AI 요약 제외)

    간략한 트렌드 정보만 빠르게 반환합니다.
    """
    # 언어 감지 및 번역
    original_language = detect_language(keyword)
    search_keyword = keyword

    if original_language == 'ko':
        search_keyword = translate_medical_terms(keyword, 'ko_to_en')

    # PubMed에서 빠르게 검색
    papers: list[dict[str, Any]] = []

    try:
        pmids = await pubmed_client.search(search_keyword, max_results=max_papers)
        if pmids:
            pubmed_papers = await pubmed_client.fetch_papers(pmids)
            for p in pubmed_papers:
                papers.append({
                    'title': p.title,
                    'abstract': p.abstract,
                    'publication_date': p.publication_date,
                })
    except Exception:
        pass

    if not papers:
        raise HTTPException(status_code=404, detail="No papers found")

    # 빠른 분석
    analyzer = TrendAnalyzer()
    analyzer.set_papers(papers)
    analyzer.analyze_publication_trend()
    analyzer.extract_key_terms(top_n=10)

    year_trend = analyzer.trend_data.get('year_trend', {})
    key_terms = analyzer.trend_data.get('key_terms', {})

    return QuickTrendResponse(
        keyword=keyword,
        total_papers=len(papers),
        years=year_trend.get('years', []),
        counts=year_trend.get('counts', []),
        top_terms=key_terms.get('terms', [])[:10],
    )


@router.post("/trends/from-papers", response_model=TrendAnalysisResponse)
async def analyze_trends_from_papers(
    papers: list[dict[str, Any]],
    keyword: str = Query(..., description="분석 키워드"),
) -> TrendAnalysisResponse:
    """
    제공된 논문 데이터로 트렌드 분석

    외부에서 수집한 논문 데이터를 직접 분석합니다.
    각 논문은 title, abstract, publication_date 필드가 필요합니다.
    """
    if not papers:
        raise HTTPException(status_code=400, detail="No papers provided")

    analyzer = TrendAnalyzer()
    analyzer.set_papers(papers)

    try:
        result = await analyzer.analyze(keyword)
        report = analyzer.generate_report(keyword)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    return TrendAnalysisResponse(
        keyword=keyword,
        original_language=detect_language(keyword),
        total_papers=result.total_papers,
        year_trend=result.year_trend,
        key_terms=result.key_terms,
        emerging_topics=result.emerging_topics,
        content_summary=result.content_summary,
        report=report,
    )
