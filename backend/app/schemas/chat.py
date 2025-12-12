from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation reference"""

    pmid: str
    title: str
    relevance_score: float = Field(..., ge=0, le=1)
    snippet: str


class ChatQuery(BaseModel):
    """Chat query request"""

    session_id: UUID | None = None
    query: str = Field(..., min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    """Chat query response"""

    session_id: UUID
    message_id: UUID
    answer: str
    citations: list[Citation]
    confidence_score: float | None = Field(None, ge=0, le=1)
    latency_ms: int


class ChatMessage(BaseModel):
    """Chat message"""

    message_id: UUID
    role: str  # user, assistant
    content: str
    citations: list[Citation] | None = None
    timestamp: datetime

    class Config:
        from_attributes = True


class ChatSessionCreate(BaseModel):
    """Create chat session"""

    title: str | None = None


class ChatSessionSummary(BaseModel):
    """Chat session summary"""

    session_id: UUID
    title: str | None
    message_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatSessionDetail(BaseModel):
    """Chat session with messages"""

    session_id: UUID
    title: str | None
    messages: list[ChatMessage]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
