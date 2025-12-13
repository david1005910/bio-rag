from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """User repository"""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(User, session)

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email"""
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id(self, user_id: UUID | str) -> User | None:
        """Get user by ID"""
        user_id_str = str(user_id) if isinstance(user_id, UUID) else user_id
        stmt = select(User).where(User.user_id == user_id_str)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_user(
        self,
        email: str,
        password_hash: str,
        name: str,
        organization: str | None = None,
        research_fields: list[str] | None = None,
        interests: list[str] | None = None,
    ) -> User:
        """Create new user"""
        user = User(
            email=email,
            password_hash=password_hash,
            name=name,
            organization=organization,
            research_fields=research_fields,
            interests=interests,
        )
        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def update_last_login(self, user_id: UUID | str) -> None:
        """Update user's last login time"""
        user = await self.get_by_id(user_id)
        if user:
            user.last_login = datetime.utcnow()
            await self.session.flush()

    async def update_interests(
        self,
        user_id: UUID | str,
        interests: list[str],
    ) -> User | None:
        """Update user's interests"""
        user = await self.get_by_id(user_id)
        if user:
            user.interests = interests[:10]  # Max 10 interests
            await self.session.flush()
            await self.session.refresh(user)
        return user

    async def deactivate(self, user_id: UUID | str) -> bool:
        """Deactivate user account"""
        user = await self.get_by_id(user_id)
        if user:
            user.is_active = False
            await self.session.flush()
            return True
        return False
