"""PubMed Direct Search API endpoints"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.pubmed.client import pubmed_client, PubMedAPIError

router = APIRouter(prefix="/pubmed", tags=["PubMed"])


# ==================== Request/Response Models ====================
class PubMedSearchQuery(BaseModel):
    """PubMed 검색 쿼리"""
    query: str = Field(..., min_length=1, max_length=500, description="검색 쿼리")
    max_results: int = Field(default=20, ge=1, le=100, description="최대 결과 수")


class PubMedPaperResponse(BaseModel):
    """PubMed 논문 응답"""
    pmid: str
    title: str
    abstract: str | None = None
    authors: list[str] | None = None
    journal: str | None = None
    publication_date: str | None = None
    doi: str | None = None
    keywords: list[str] | None = None
    mesh_terms: list[str] | None = None
    pdf_url: str | None = None


class PubMedSearchResult(BaseModel):
    """PubMed 검색 결과"""
    results: list[PubMedPaperResponse]
    total: int
    query: str


# ==================== Endpoints ====================
@router.post("/search", response_model=PubMedSearchResult)
async def search_pubmed(search_query: PubMedSearchQuery) -> PubMedSearchResult:
    """
    PubMed 실시간 검색

    - **query**: 검색 쿼리 (예: "diabetes treatment", "CRISPR gene editing")
    - **max_results**: 최대 결과 수 (1-100)

    NCBI E-utilities API를 사용하여 PubMed를 직접 검색합니다.
    """
    try:
        # Search for PMIDs
        pmids = await pubmed_client.search(
            query=search_query.query,
            max_results=search_query.max_results,
        )

        if not pmids:
            return PubMedSearchResult(
                results=[],
                total=0,
                query=search_query.query,
            )

        # Fetch paper details
        papers = await pubmed_client.fetch_papers(pmids)

        results = [
            PubMedPaperResponse(
                pmid=p.pmid,
                title=p.title,
                abstract=p.abstract,
                authors=p.authors,
                journal=p.journal,
                publication_date=p.publication_date.isoformat() if p.publication_date else None,
                doi=p.doi,
                keywords=p.keywords,
                mesh_terms=p.mesh_terms,
                pdf_url=f"https://pubmed.ncbi.nlm.nih.gov/{p.pmid}/" if p.pmid else None,
            )
            for p in papers
        ]

        return PubMedSearchResult(
            results=results,
            total=len(results),
            query=search_query.query,
        )

    except PubMedAPIError as e:
        raise HTTPException(status_code=503, detail=f"PubMed API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/search", response_model=PubMedSearchResult)
async def search_pubmed_get(
    query: str = Query(..., min_length=1, max_length=500, description="검색 쿼리"),
    max_results: int = Query(default=20, ge=1, le=100, description="최대 결과 수"),
) -> PubMedSearchResult:
    """
    PubMed 실시간 검색 (GET)
    """
    return await search_pubmed(PubMedSearchQuery(query=query, max_results=max_results))


@router.get("/paper/{pmid}", response_model=PubMedPaperResponse)
async def get_pubmed_paper(pmid: str) -> PubMedPaperResponse:
    """
    PubMed 논문 상세 조회

    - **pmid**: PubMed ID

    지정된 PMID의 논문 상세 정보를 반환합니다.
    """
    try:
        papers = await pubmed_client.fetch_papers([pmid])

        if not papers:
            raise HTTPException(status_code=404, detail=f"Paper not found: {pmid}")

        p = papers[0]
        return PubMedPaperResponse(
            pmid=p.pmid,
            title=p.title,
            abstract=p.abstract,
            authors=p.authors,
            journal=p.journal,
            publication_date=p.publication_date.isoformat() if p.publication_date else None,
            doi=p.doi,
            keywords=p.keywords,
            mesh_terms=p.mesh_terms,
            pdf_url=f"https://pubmed.ncbi.nlm.nih.gov/{p.pmid}/",
        )

    except PubMedAPIError as e:
        raise HTTPException(status_code=503, detail=f"PubMed API error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch failed: {str(e)}")
