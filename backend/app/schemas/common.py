from typing import Any

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Error detail information"""

    code: str
    message: str
    details: dict[str, Any] | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response"""

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    version: str


class PaginatedResponse(BaseModel):
    """Base paginated response"""

    total: int
    limit: int
    offset: int
