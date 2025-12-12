from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.chat import ChatMessage, ChatSession
from app.repositories.base import BaseRepository


class ChatRepository(BaseRepository[ChatSession]):
    """Chat repository for sessions and messages"""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(ChatSession, session)

    async def get_session(self, session_id: UUID) -> ChatSession | None:
        """Get chat session by ID"""
        stmt = select(ChatSession).where(ChatSession.session_id == session_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_session_with_messages(
        self,
        session_id: UUID,
    ) -> ChatSession | None:
        """Get chat session with messages"""
        stmt = (
            select(ChatSession)
            .options(selectinload(ChatSession.messages))
            .where(ChatSession.session_id == session_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_session(
        self,
        user_id: UUID,
        title: str | None = None,
    ) -> ChatSession:
        """Create new chat session"""
        chat_session = ChatSession(
            user_id=user_id,
            title=title,
        )
        self.session.add(chat_session)
        await self.session.flush()
        await self.session.refresh(chat_session)
        return chat_session

    async def list_sessions(
        self,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[ChatSession], int]:
        """List user's chat sessions"""
        # Count
        count_stmt = (
            select(func.count())
            .select_from(ChatSession)
            .where(ChatSession.user_id == user_id)
        )
        count_result = await self.session.execute(count_stmt)
        total = count_result.scalar() or 0

        # Data
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        sessions = list(result.scalars().all())

        return sessions, total

    async def update_session_title(
        self,
        session_id: UUID,
        title: str,
    ) -> ChatSession | None:
        """Update session title"""
        chat_session = await self.get_session(session_id)
        if chat_session:
            chat_session.title = title
            chat_session.updated_at = datetime.utcnow()
            await self.session.flush()
            await self.session.refresh(chat_session)
        return chat_session

    async def delete_session(self, session_id: UUID) -> bool:
        """Delete chat session and all messages"""
        chat_session = await self.get_session(session_id)
        if chat_session:
            await self.session.delete(chat_session)
            await self.session.flush()
            return True
        return False

    async def add_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        latency_ms: int | None = None,
    ) -> ChatMessage:
        """Add message to session"""
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            citations=citations,
            latency_ms=latency_ms,
        )
        self.session.add(message)

        # Update session timestamp
        chat_session = await self.get_session(session_id)
        if chat_session:
            chat_session.updated_at = datetime.utcnow()

            # Auto-generate title from first user message if not set
            if chat_session.title is None and role == "user":
                chat_session.title = content[:50] + ("..." if len(content) > 50 else "")

        await self.session.flush()
        await self.session.refresh(message)
        return message

    async def get_session_messages(
        self,
        session_id: UUID,
        limit: int = 50,
    ) -> list[ChatMessage]:
        """Get messages for a session"""
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_message_count(self, session_id: UUID) -> int:
        """Get message count for a session"""
        stmt = (
            select(func.count())
            .select_from(ChatMessage)
            .where(ChatMessage.session_id == session_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0
