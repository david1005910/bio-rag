from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Table, Integer, TypeDecorator
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from .database import Base

class GUID(TypeDecorator):
    """Platform-independent GUID type that works with SQLite and PostgreSQL."""
    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            if isinstance(value, uuid.UUID):
                return str(value)
            return str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
        return value

user_saved_papers = Table(
    'user_saved_papers',
    Base.metadata,
    Column('user_id', GUID(), ForeignKey('users.id'), primary_key=True),
    Column('paper_pmid', String(20), ForeignKey('papers.pmid'), primary_key=True),
    Column('saved_at', DateTime, default=datetime.utcnow),
    Column('notes', Text)
)

class User(Base):
    __tablename__ = "users"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255))
    password_hash = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

    saved_papers = relationship("Paper", secondary=user_saved_papers)
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False)
    title = Column(String(255), default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id = Column(GUID(), ForeignKey('chat_sessions.id'), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    sources = Column(Text)  # JSON string of source papers
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")

class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey('users.id'))
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50))
    response_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
