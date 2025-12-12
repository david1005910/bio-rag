"""Authentication service"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.user import UserRepository
from app.schemas.user import Token, UserCreate, UserResponse
from app.services.auth.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
    verify_token_type,
)

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Authentication error"""

    pass


class AuthService:
    """Authentication service"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.user_repo = UserRepository(db)

    async def register(self, user_data: UserCreate) -> UserResponse:
        """
        Register a new user

        Args:
            user_data: User registration data

        Returns:
            Created user

        Raises:
            AuthenticationError: If email already exists
        """
        # Check if email exists
        existing = await self.user_repo.get_by_email(user_data.email)
        if existing:
            raise AuthenticationError("Email already registered")

        # Hash password and create user
        hashed_password = hash_password(user_data.password)

        user = await self.user_repo.create(
            email=user_data.email,
            password_hash=hashed_password,
            name=user_data.name,
            organization=user_data.organization,
            research_fields=user_data.research_fields,
            interests=user_data.interests,
        )

        logger.info(f"User registered: {user.email}")
        return UserResponse.model_validate(user)

    async def login(self, email: str, password: str) -> Token:
        """
        Authenticate user and return tokens

        Args:
            email: User email
            password: User password

        Returns:
            JWT tokens

        Raises:
            AuthenticationError: If credentials are invalid
        """
        user = await self.user_repo.get_by_email(email)
        if not user:
            raise AuthenticationError("Invalid email or password")

        if not verify_password(password, user.password_hash):
            raise AuthenticationError("Invalid email or password")

        if not user.is_active:
            raise AuthenticationError("User account is inactive")

        # Update last login
        await self.user_repo.update(user.user_id, last_login=datetime.now(timezone.utc))

        # Generate tokens
        access_token = create_access_token(str(user.user_id))
        refresh_token = create_refresh_token(str(user.user_id))

        logger.info(f"User logged in: {user.email}")

        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
        )

    async def refresh_tokens(self, refresh_token: str) -> Token:
        """
        Refresh access token

        Args:
            refresh_token: Valid refresh token

        Returns:
            New tokens

        Raises:
            AuthenticationError: If token is invalid
        """
        payload = decode_token(refresh_token)
        if not payload:
            raise AuthenticationError("Invalid refresh token")

        if not verify_token_type(payload, "refresh"):
            raise AuthenticationError("Invalid token type")

        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationError("Invalid token payload")

        # Verify user exists and is active
        user = await self.user_repo.get(UUID(user_id))
        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")

        # Generate new tokens
        new_access = create_access_token(user_id)
        new_refresh = create_refresh_token(user_id)

        return Token(
            access_token=new_access,
            refresh_token=new_refresh,
        )

    async def get_current_user(self, token: str) -> User:
        """
        Get current user from access token

        Args:
            token: Access token

        Returns:
            Current user

        Raises:
            AuthenticationError: If token is invalid
        """
        payload = decode_token(token)
        if not payload:
            raise AuthenticationError("Invalid access token")

        if not verify_token_type(payload, "access"):
            raise AuthenticationError("Invalid token type")

        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationError("Invalid token payload")

        user = await self.user_repo.get(UUID(user_id))
        if not user:
            raise AuthenticationError("User not found")

        if not user.is_active:
            raise AuthenticationError("User account is inactive")

        return user
