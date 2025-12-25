from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Any
from datetime import datetime
from uuid import UUID

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=8, description="Password must be exactly 8 characters")
    name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: UUID
    email: str
    name: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    limit: int = Field(10, ge=1, le=100)
    date_from: Optional[str] = None
    date_to: Optional[str] = None

class PaperResponse(BaseModel):
    pmid: str
    title: str
    abstract: Optional[str]
    authors: List[str]
    journal: Optional[str]
    publication_date: Optional[str]
    doi: Optional[str]
    keywords: List[str]
    relevance: Optional[float] = None
    excerpt: Optional[str] = None

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: Optional[UUID] = None
    reasoning_mode: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float
    session_id: UUID
    message_id: UUID
    reasoning_steps: Optional[List[dict]] = None

class ChatSessionResponse(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatMessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    sources: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class IndexPaperRequest(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: List[str] = []
    journal: Optional[str] = None
    publication_date: Optional[str] = None

class TrendKeyword(BaseModel):
    keyword: str
    count: int
    growth_rate: Optional[float] = None

class TrendResponse(BaseModel):
    period: str
    keywords: List[TrendKeyword]

class SimilarPaperRequest(BaseModel):
    pmid: str
    limit: int = Field(5, ge=1, le=20)

class ErrorResponse(BaseModel):
    detail: str
