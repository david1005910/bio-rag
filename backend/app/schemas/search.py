from pydantic import BaseModel, Field

from app.schemas.paper import PaperSummary


class SearchFilters(BaseModel):
    """Search filter options"""

    year_from: int | None = None
    year_to: int | None = None
    journals: list[str] | None = None
    authors: list[str] | None = None


class SearchQuery(BaseModel):
    """Search query request"""

    query: str = Field(..., min_length=1, max_length=500)
    filters: SearchFilters | None = None
    limit: int = Field(default=10, ge=1, le=50)
    offset: int = Field(default=0, ge=0)


class SearchResult(BaseModel):
    """Search result response"""

    results: list[PaperSummary]
    total: int
    query_time_ms: int


class SearchSuggestion(BaseModel):
    """Search suggestion"""

    text: str
    score: float
