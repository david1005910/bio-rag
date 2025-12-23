"""ArXiv API endpoints"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.arxiv import ArXivAPIError, ArXivPaper, arxiv_client

router = APIRouter(prefix="/arxiv", tags=["ArXiv"])


# ==================== Request/Response Models ====================
class ArXivSearchQuery(BaseModel):
    """ArXiv 검색 요청"""
    query: str = Field(..., min_length=1, max_length=500, description="검색 쿼리")
    max_results: int = Field(default=10, ge=1, le=100, description="최대 결과 수")
    sort_by: str = Field(default="relevance", description="정렬 기준")


class ArXivPaperResponse(BaseModel):
    """ArXiv 논문 응답"""
    arxiv_id: str
    title: str
    abstract: str | None = None
    authors: list[str] | None = None
    published: datetime | None = None
    pdf_url: str | None = None
    categories: list[str] | None = None
    doi: str | None = None

    class Config:
        from_attributes = True


class ArXivSearchResult(BaseModel):
    """ArXiv 검색 결과"""
    results: list[ArXivPaperResponse]
    total: int
    query: str


# ==================== Endpoints ====================
@router.post("/search", response_model=ArXivSearchResult)
async def search_arxiv(search_query: ArXivSearchQuery) -> ArXivSearchResult:
    """
    arXiv 논문 검색

    - **query**: 검색 쿼리 (예: "machine learning", "diabetes treatment")
    - **max_results**: 최대 결과 수 (1-100)
    - **sort_by**: 정렬 기준 (relevance, lastUpdatedDate, submittedDate)

    arXiv API를 사용하여 과학/공학 논문을 검색합니다.
    """
    try:
        papers = await arxiv_client.search(
            query=search_query.query,
            max_results=search_query.max_results,
            sort_by=search_query.sort_by,
        )

        return ArXivSearchResult(
            results=[ArXivPaperResponse.model_validate(p) for p in papers],
            total=len(papers),
            query=search_query.query,
        )

    except ArXivAPIError as e:
        raise HTTPException(status_code=503, detail=f"ArXiv API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paper/{arxiv_id}", response_model=ArXivPaperResponse)
async def get_arxiv_paper(arxiv_id: str) -> ArXivPaperResponse:
    """
    특정 arXiv 논문 조회

    - **arxiv_id**: arXiv ID (예: "2301.00001")

    논문 메타데이터와 PDF URL을 반환합니다.
    """
    try:
        paper = await arxiv_client.fetch_paper(arxiv_id)

        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found: {arxiv_id}")

        return ArXivPaperResponse.model_validate(paper)

    except ArXivAPIError as e:
        raise HTTPException(status_code=503, detail=f"ArXiv API error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=ArXivSearchResult)
async def search_arxiv_get(
    query: str = Query(..., min_length=1, max_length=500, description="검색 쿼리"),
    max_results: int = Query(default=10, ge=1, le=100, description="최대 결과 수"),
    sort_by: str = Query(default="relevance", description="정렬 기준"),
) -> ArXivSearchResult:
    """
    arXiv 논문 검색 (GET)

    POST /arxiv/search 와 동일한 기능의 GET 엔드포인트
    """
    return await search_arxiv(
        ArXivSearchQuery(query=query, max_results=max_results, sort_by=sort_by)
    )
