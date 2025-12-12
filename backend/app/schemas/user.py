from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """Base user schema"""

    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)
    organization: str | None = None
    research_fields: list[str] | None = None
    interests: list[str] | None = Field(None, max_length=10)


class UserCreate(UserBase):
    """Schema for creating a user"""

    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    """Schema for updating a user"""

    name: str | None = Field(None, min_length=1, max_length=100)
    organization: str | None = None
    research_fields: list[str] | None = None
    interests: list[str] | None = Field(None, max_length=10)


class UserResponse(UserBase):
    """Schema for user response"""

    user_id: UUID
    subscription_tier: str
    created_at: datetime
    last_login: datetime | None = None
    is_active: bool

    class Config:
        from_attributes = True


class UserProfile(BaseModel):
    """Schema for user profile with usage info"""

    user_id: UUID
    email: str
    name: str
    organization: str | None
    research_fields: list[str] | None
    interests: list[str] | None
    subscription_tier: str
    usage: "UsageInfo"

    class Config:
        from_attributes = True


class UsageInfo(BaseModel):
    """User usage information"""

    queries_this_month: int
    queries_limit: int | None = None


class Token(BaseModel):
    """JWT token response"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    """JWT token payload"""

    sub: str
    exp: datetime
