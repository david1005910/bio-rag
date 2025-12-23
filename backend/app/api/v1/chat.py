"""Chat API endpoints"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, DbSession
from app.core.config import settings
from app.schemas.chat import (
    ChatQuery,
    ChatResponse,
    ChatSessionCreate,
    ChatSessionDetail,
    ChatSessionSummary,
)
from app.services.chat.service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


def _check_openai_configured() -> None:
    """Check if OpenAI API key is configured"""
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your-"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not configured. Please set a valid OPENAI_API_KEY.",
        )


@router.post("/sessions", response_model=ChatSessionSummary, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_data: ChatSessionCreate,
    current_user: CurrentUser,
    db: DbSession,
) -> ChatSessionSummary:
    """
    Create a new chat session

    - **title**: Optional session title

    Requires authentication.
    """
    chat_service = ChatService(db)
    return await chat_service.create_session(
        user_id=current_user.user_id,
        title=session_data.title,
    )


@router.get("/sessions", response_model=list[ChatSessionSummary])
async def list_sessions(
    current_user: CurrentUser,
    db: DbSession,
    limit: int = 20,
    offset: int = 0,
) -> list[ChatSessionSummary]:
    """
    List user's chat sessions

    - **limit**: Number of sessions (1-100, default 20)
    - **offset**: Pagination offset

    Requires authentication.
    """
    chat_service = ChatService(db)
    return await chat_service.get_sessions(
        user_id=current_user.user_id,
        limit=min(limit, 100),
        offset=offset,
    )


@router.get("/sessions/{session_id}", response_model=ChatSessionDetail)
async def get_session(
    session_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
) -> ChatSessionDetail:
    """
    Get a chat session with messages

    - **session_id**: Session ID

    Requires authentication.
    """
    chat_service = ChatService(db)
    session = await chat_service.get_session(
        session_id=session_id,
        user_id=current_user.user_id,
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    return session


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
) -> None:
    """
    Delete a chat session

    - **session_id**: Session ID

    This action is irreversible.
    Requires authentication.
    """
    chat_service = ChatService(db)
    success = await chat_service.delete_session(
        session_id=session_id,
        user_id=current_user.user_id,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )


@router.post("/query", response_model=ChatResponse)
async def query(
    chat_query: ChatQuery,
    current_user: CurrentUser,
    db: DbSession,
) -> ChatResponse:
    """
    Ask a question using RAG

    - **session_id**: Optional session ID (creates new session if not provided)
    - **query**: Question to ask (natural language)

    Returns answer with citations from relevant papers.
    Requires authentication.
    """
    import uuid

    # Check if demo mode is enabled
    if settings.DEMO_MODE:
        import time
        start_time = time.time()
        from app.services.demo import get_demo_chat_response
        demo_result = get_demo_chat_response(chat_query.query)
        latency = int((time.time() - start_time) * 1000)
        return ChatResponse(
            session_id=chat_query.session_id or uuid.uuid4(),
            message_id=uuid.uuid4(),
            answer=demo_result["answer"],
            citations=[
                {
                    "pmid": c["pmid"],
                    "title": c["title"],
                    "relevance_score": c["relevance_score"],
                    "snippet": c.get("snippet", "Relevant research excerpt from this paper."),
                }
                for c in demo_result["citations"]
            ],
            latency_ms=latency,
        )

    _check_openai_configured()

    chat_service = ChatService(db)
    try:
        return await chat_service.query(
            user_id=current_user.user_id,
            chat_query=chat_query,
        )
    except ValueError as e:
        error_msg = str(e)
        if "Embedding function not set" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chat service not initialized. Please check server configuration.",
            )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/query/stream")
async def query_stream(
    chat_query: ChatQuery,
    current_user: CurrentUser,
    db: DbSession,
) -> StreamingResponse:
    """
    Ask a question with streaming response

    - **session_id**: Optional session ID
    - **query**: Question to ask

    Returns streaming response as Server-Sent Events.
    Requires authentication.
    """
    _check_openai_configured()

    chat_service = ChatService(db)

    # Detect language
    language = chat_service._detect_language(chat_query.query)

    async def stream_response():
        try:
            async for chunk in chat_service.rag.stream(
                query=chat_query.query,
                language=language,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
