"""Search API endpoints"""

from fastapi import APIRouter, HTTPException

from app.api.deps import DbSession, OptionalUser
from app.core.config import settings
from app.schemas.paper import PaperSummary
from app.schemas.search import SearchQuery, SearchResult
from app.services.search.service import SearchService

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResult)
async def search_papers(
    search_query: SearchQuery,
    db: DbSession,
    current_user: OptionalUser,
) -> SearchResult:
    """
    Search for papers using semantic search

    - **query**: Search query (natural language or keywords)
    - **filters**: Optional filters (year range, journals, authors)
    - **limit**: Number of results (1-50, default 10)
    - **offset**: Pagination offset

    Uses hybrid search (dense + BM25) with cross-encoder reranking.
    Authentication is optional but allows search history tracking.
    """
    # Check if demo mode is enabled
    if settings.DEMO_MODE:
        from app.services.demo import get_demo_search_results
        demo_result = get_demo_search_results(search_query.query, search_query.limit)
        return SearchResult(
            results=[
                PaperSummary(
                    pmid=p["pmid"],
                    title=p["title"],
                    authors=p["authors"],
                    journal=p["journal"],
                    publication_date=p["publication_date"],
                    abstract=p["abstract"],
                    relevance_score=p["relevance_score"],
                )
                for p in demo_result["results"]
            ],
            total=demo_result["total"],
            query_time_ms=demo_result["query_time_ms"],
        )

    # Check if OpenAI API key is configured
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your-"):
        raise HTTPException(
            status_code=503,
            detail="Search service not configured. Please set a valid OPENAI_API_KEY.",
        )

    search_service = SearchService(db)

    try:
        result = await search_service.search(
            query=search_query.query,
            filters=search_query.filters,
            limit=search_query.limit,
            offset=search_query.offset,
            user_id=current_user.user_id if current_user else None,
        )
    except ValueError as e:
        error_msg = str(e)
        if "Embedding function not set" in error_msg:
            raise HTTPException(
                status_code=503,
                detail="Search service not initialized. Please check server configuration.",
            )
        raise HTTPException(status_code=500, detail=str(e))

    return result


@router.get("/similar/{pmid}", response_model=list[PaperSummary])
async def get_similar_papers(
    pmid: str,
    limit: int = 5,
    db: DbSession = None,
) -> list[PaperSummary]:
    """
    Get papers similar to a given paper

    - **pmid**: Paper PMID
    - **limit**: Number of similar papers (1-20, default 5)

    Returns papers with similar content based on semantic similarity.
    """
    search_service = SearchService(db)

    return await search_service.get_similar_papers(
        pmid=pmid,
        limit=min(limit, 20),
    )


@router.post("/by-pmids", response_model=list[PaperSummary])
async def get_papers_by_pmids(
    pmids: list[str],
    db: DbSession,
) -> list[PaperSummary]:
    """
    Get papers by their PMIDs

    - **pmids**: List of PMIDs

    Returns paper summaries for the given PMIDs.
    """
    search_service = SearchService(db)

    return await search_service.search_by_pmids(pmids[:100])  # Limit to 100
