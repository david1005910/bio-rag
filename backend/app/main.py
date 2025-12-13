from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import api_router
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
# Import all models to register them with SQLAlchemy
from app.models import User, Paper, SavedPaper, ChatSession, ChatMessage, SearchLog


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    # Startup
    print(f"Starting up {settings.APP_NAME} API...")
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created/verified.")

    # Initialize embedding function for search
    if settings.OPENAI_API_KEY:
        try:
            from app.services.rag.embedding import embed_text
            from app.services.rag.hybrid_search import hybrid_searcher
            hybrid_searcher.set_embedding_func(embed_text)
            print("Embedding function initialized for hybrid search.")
        except Exception as e:
            print(f"Warning: Failed to initialize embedding function: {e}")
    else:
        print("Warning: OPENAI_API_KEY not set. Search functionality will be limited.")

    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="Bio-RAG API",
    description="Biomedical Research AI-Guided Analytics Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint"""
    return {"message": "Welcome to Bio-RAG API"}


# Include API router
app.include_router(api_router, prefix="/api/v1")
