from sqlalchemy import Column, String, Text, DateTime, Integer, Table, ForeignKey, TypeDecorator
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

paper_authors = Table(
    'paper_authors',
    Base.metadata,
    Column('paper_pmid', String(20), ForeignKey('papers.pmid'), primary_key=True),
    Column('author_id', Integer, ForeignKey('authors.id'), primary_key=True),
    Column('author_order', Integer)
)

paper_keywords = Table(
    'paper_keywords',
    Base.metadata,
    Column('paper_pmid', String(20), ForeignKey('papers.pmid'), primary_key=True),
    Column('keyword_id', Integer, ForeignKey('keywords.id'), primary_key=True)
)

class Paper(Base):
    __tablename__ = "papers"

    pmid = Column(String(20), primary_key=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    full_text = Column(Text)
    doi = Column(String(100))
    journal = Column(String(255))
    publication_date = Column(DateTime)
    citation_count = Column(Integer, default=0)
    pdf_url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    authors = relationship("Author", secondary=paper_authors, back_populates="papers")
    keywords = relationship("Keyword", secondary=paper_keywords, back_populates="papers")
    chunks = relationship("Chunk", back_populates="paper", cascade="all, delete-orphan")

class Author(Base):
    __tablename__ = "authors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    affiliation = Column(Text)
    email = Column(String(255))

    papers = relationship("Paper", secondary=paper_authors, back_populates="authors")

class Keyword(Base):
    __tablename__ = "keywords"

    id = Column(Integer, primary_key=True, autoincrement=True)
    term = Column(String(255), unique=True, nullable=False)
    type = Column(String(50))

    papers = relationship("Paper", secondary=paper_keywords, back_populates="keywords")

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    paper_pmid = Column(String(20), ForeignKey('papers.pmid'), nullable=False)
    section = Column(String(50))
    text = Column(Text, nullable=False)
    chunk_index = Column(Integer)
    token_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    paper = relationship("Paper", back_populates="chunks")
