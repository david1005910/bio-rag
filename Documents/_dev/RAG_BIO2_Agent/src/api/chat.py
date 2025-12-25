from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
import json

from src.models import get_db, User, ChatSession, ChatMessage
from src.services import RAGService
from .schemas import ChatRequest, ChatResponse, ChatSessionResponse, ChatMessageResponse
from .auth import get_current_user

router = APIRouter(prefix="/chat", tags=["Chat"])

_rag_service = None

def get_rag_service():
    global _rag_service
    if _rag_service is None:
        # Try simple embeddings first (works without external dependencies)
        # For production, use pubmedbert or openai with proper API keys
        try:
            _rag_service = RAGService(
                embedding_model="simple",
                vector_dimension=768,
                collection_name="biomedical_papers_768d"
            )
        except Exception as e:
            print(f"Failed to initialize RAG service: {e}")
            raise
    return _rag_service

@router.post("/query", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        if request.session_id:
            session = db.query(ChatSession).filter(
                ChatSession.id == request.session_id,
                ChatSession.user_id == current_user.id
            ).first()
            
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chat session not found"
                )
        else:
            session = ChatSession(
                user_id=current_user.id,
                title=request.question[:50] + "..." if len(request.question) > 50 else request.question
            )
            db.add(session)
            db.commit()
            db.refresh(session)
        
        user_message = ChatMessage(
            session_id=session.id,
            role="user",
            content=request.question
        )
        db.add(user_message)
        db.commit()
        
        rag_service = get_rag_service()
        if request.reasoning_mode:
            response = await rag_service.reasoning_query(request.question, top_k=5)
        else:
            response = await rag_service.query(request.question, top_k=5)
        
        assistant_message = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=response.answer,
            sources=json.dumps(response.sources)
        )
        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)
        
        return ChatResponse(
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            session_id=session.id,
            message_id=assistant_message.id,
            reasoning_steps=response.reasoning_steps if request.reasoning_mode else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat query failed: {str(e)}"
        )

@router.get("/sessions", response_model=List[ChatSessionResponse])
async def get_chat_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == current_user.id
    ).order_by(ChatSession.updated_at.desc()).all()
    
    return [
        ChatSessionResponse(
            id=s.id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at
        )
        for s in sessions
    ]

@router.get("/sessions/{session_id}", response_model=List[ChatMessageResponse])
async def get_session_messages(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )
    
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at.asc()).all()
    
    return [
        ChatMessageResponse(
            id=m.id,
            role=m.role,
            content=m.content,
            sources=m.sources,
            created_at=m.created_at
        )
        for m in messages
    ]

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )
    
    db.delete(session)
    db.commit()
    
    return {"status": "success", "message": "Session deleted"}
