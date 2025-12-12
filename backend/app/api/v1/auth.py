"""Authentication API endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from app.api.deps import DbSession
from app.schemas.user import Token, UserCreate, UserResponse
from app.services.auth.service import AuthenticationError, AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    """Login request body"""

    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    """Refresh token request body"""

    refresh_token: str


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: DbSession,
) -> UserResponse:
    """
    Register a new user

    - **email**: Valid email address
    - **password**: Password (min 8 characters)
    - **name**: User's name
    - **organization**: Optional organization
    - **research_fields**: Optional list of research fields
    - **interests**: Optional list of interests
    """
    auth_service = AuthService(db)

    try:
        user = await auth_service.register(user_data)
        return user
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    db: DbSession,
) -> Token:
    """
    Authenticate user and get access tokens

    - **email**: User email
    - **password**: User password

    Returns access and refresh tokens.
    """
    auth_service = AuthService(db)

    try:
        tokens = await auth_service.login(login_data.email, login_data.password)
        return tokens
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )


@router.post("/refresh", response_model=Token)
async def refresh_tokens(
    refresh_data: RefreshRequest,
    db: DbSession,
) -> Token:
    """
    Refresh access token using refresh token

    - **refresh_token**: Valid refresh token

    Returns new access and refresh tokens.
    """
    auth_service = AuthService(db)

    try:
        tokens = await auth_service.refresh_tokens(refresh_data.refresh_token)
        return tokens
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )
