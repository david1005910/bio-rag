from datetime import datetime

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Full paper metadata schema"""

    pmid: str
    title: str
    abstract: str | None = None
    authors: list[str] | None = None
    journal: str | None = None
    publication_date: datetime | None = None
    doi: str | None = None
    keywords: list[str] | None = None
    mesh_terms: list[str] | None = None
    citation_count: int | None = None
    pdf_url: str | None = None

    class Config:
        from_attributes = True


class PaperSummary(BaseModel):
    """Paper summary for search results"""

    pmid: str
    title: str
    abstract: str | None = Field(None, description="Truncated to 300 chars")
    authors: list[str] | None = None
    journal: str | None = None
    publication_date: datetime | None = None
    relevance_score: float = Field(..., ge=0, le=100)
    pdf_url: str | None = None

    class Config:
        from_attributes = True


class PaperDetail(PaperMetadata):
    """Paper detail with similar papers"""

    similar_papers: list[PaperSummary] | None = None
    created_at: datetime
    updated_at: datetime


class SavedPaperCreate(BaseModel):
    """Schema for saving a paper"""

    pmid: str
    tags: list[str] | None = None
    notes: str | None = None


class SavedPaperResponse(BaseModel):
    """Schema for saved paper response"""

    pmid: str
    title: str
    saved_at: datetime
    tags: list[str] | None = None
    notes: str | None = None

    class Config:
        from_attributes = True
