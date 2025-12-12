"""Chat service for RAG-based Q&A"""

import logging
import time
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatMessage as ChatMessageModel
from app.models.chat import ChatSession
from app.repositories.chat import ChatRepository
from app.schemas.chat import (
    ChatMessage,
    ChatQuery,
    ChatResponse,
    ChatSessionDetail,
    ChatSessionSummary,
    Citation,
)
from app.services.rag.chain import RAGChain, RAGConfig, rag_chain

logger = logging.getLogger(__name__)


class ChatService:
    """Service for RAG-based chat"""

    def __init__(
        self,
        db: AsyncSession,
        rag_chain: RAGChain | None = None,
    ) -> None:
        self.db = db
        self.chat_repo = ChatRepository(db)
        self._rag_chain = rag_chain

    @property
    def rag(self) -> RAGChain:
        """Get RAG chain"""
        if self._rag_chain is None:
            self._rag_chain = rag_chain
        return self._rag_chain

    async def create_session(
        self,
        user_id: UUID,
        title: str | None = None,
    ) -> ChatSessionSummary:
        """
        Create a new chat session

        Args:
            user_id: User ID
            title: Optional session title

        Returns:
            Created session summary
        """
        session = await self.chat_repo.create_session(
            user_id=user_id,
            title=title,
        )

        return ChatSessionSummary(
            session_id=session.session_id,
            title=session.title,
            message_count=0,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    async def get_sessions(
        self,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[ChatSessionSummary]:
        """
        Get user's chat sessions

        Args:
            user_id: User ID
            limit: Number of sessions
            offset: Pagination offset

        Returns:
            List of session summaries
        """
        sessions = await self.chat_repo.get_user_sessions(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

        summaries: list[ChatSessionSummary] = []
        for session in sessions:
            message_count = await self.chat_repo.get_message_count(session.session_id)
            summaries.append(
                ChatSessionSummary(
                    session_id=session.session_id,
                    title=session.title,
                    message_count=message_count,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                )
            )

        return summaries

    async def get_session(
        self,
        session_id: UUID,
        user_id: UUID,
    ) -> ChatSessionDetail | None:
        """
        Get chat session with messages

        Args:
            session_id: Session ID
            user_id: User ID (for ownership verification)

        Returns:
            Session detail or None if not found
        """
        session = await self.chat_repo.get_session(session_id)
        if not session or session.user_id != user_id:
            return None

        messages = await self.chat_repo.get_session_messages(session_id)

        return ChatSessionDetail(
            session_id=session.session_id,
            title=session.title,
            messages=[self._convert_message(msg) for msg in messages],
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    async def delete_session(
        self,
        session_id: UUID,
        user_id: UUID,
    ) -> bool:
        """
        Delete a chat session

        Args:
            session_id: Session ID
            user_id: User ID (for ownership verification)

        Returns:
            True if deleted, False if not found
        """
        session = await self.chat_repo.get_session(session_id)
        if not session or session.user_id != user_id:
            return False

        return await self.chat_repo.delete_session(session_id)

    async def query(
        self,
        user_id: UUID,
        chat_query: ChatQuery,
    ) -> ChatResponse:
        """
        Process a chat query using RAG

        Args:
            user_id: User ID
            chat_query: Chat query

        Returns:
            Chat response with answer and citations
        """
        start_time = time.time()

        # Get or create session
        session_id = chat_query.session_id
        if not session_id:
            session = await self.chat_repo.create_session(
                user_id=user_id,
                title=chat_query.query[:50],  # Use first 50 chars as title
            )
            session_id = session.session_id
        else:
            # Verify session ownership
            session = await self.chat_repo.get_session(session_id)
            if not session or session.user_id != user_id:
                # Create new session if not found or not owned
                session = await self.chat_repo.create_session(
                    user_id=user_id,
                    title=chat_query.query[:50],
                )
                session_id = session.session_id

        # Store user message
        user_message_id = uuid4()
        await self.chat_repo.add_message(
            session_id=session_id,
            message_id=user_message_id,
            role="user",
            content=chat_query.query,
        )

        # Detect language from query
        language = self._detect_language(chat_query.query)

        # Invoke RAG chain
        rag_response = await self.rag.invoke(
            query=chat_query.query,
            language=language,
        )

        # Convert sources to citations
        citations = self._convert_to_citations(rag_response)

        # Store assistant message
        assistant_message_id = uuid4()
        await self.chat_repo.add_message(
            session_id=session_id,
            message_id=assistant_message_id,
            role="assistant",
            content=rag_response.answer,
            citations=[c.model_dump() for c in citations],
        )

        # Update session
        await self.chat_repo.update_session(session_id)

        latency_ms = int((time.time() - start_time) * 1000)

        return ChatResponse(
            session_id=session_id,
            message_id=assistant_message_id,
            answer=rag_response.answer,
            citations=citations,
            confidence_score=rag_response.validation.confidence_score,
            latency_ms=latency_ms,
        )

    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        # Check for Korean characters
        for char in text:
            if '\uac00' <= char <= '\ud7a3':  # Korean syllables range
                return "ko"
        return "en"

    def _convert_to_citations(self, rag_response) -> list[Citation]:
        """Convert RAG response sources to citations"""
        citations: list[Citation] = []
        seen_pmids: set[str] = set()

        for source in rag_response.sources:
            if source.pmid in seen_pmids:
                continue
            seen_pmids.add(source.pmid)

            citations.append(
                Citation(
                    pmid=source.pmid,
                    title=source.title,
                    relevance_score=source.score,
                    snippet=source.content[:200] if source.content else "",
                )
            )

        return citations[:10]  # Limit to 10 citations

    def _convert_message(self, message: ChatMessageModel) -> ChatMessage:
        """Convert database message to schema"""
        citations = None
        if message.citations:
            citations = [Citation(**c) for c in message.citations]

        return ChatMessage(
            message_id=message.message_id,
            role=message.role,
            content=message.content,
            citations=citations,
            timestamp=message.created_at,
        )
