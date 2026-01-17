# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bio-RAG (Biomedical Research AI-Guided Analytics) is an AI-powered biomedical research paper analysis platform. It uses RAG technology to help researchers search, analyze, and extract insights from scientific literature via PubMed integration and semantic vector search.

## Development Commands

```bash
# Install dependencies (using UV package manager)
uv sync

# Run the development server (starts on http://0.0.0.0:5000)
python main.py
# Or with auto-reload:
uvicorn main:app --reload --host 0.0.0.0 --port 5000

# Database tables are auto-created on startup via SQLAlchemy lifespan manager

# Run tests (install test deps first: uv sync --extra test)
pytest                          # Run all tests
pytest tests/test_rag_service.py  # Run RAG service tests
pytest -k "test_query"          # Run tests matching pattern
pytest -v                       # Verbose output

# Test coverage
pytest --cov=src --cov-report=term-missing   # Terminal report
pytest --cov=src --cov-report=html           # HTML report in htmlcov/

# Linting (CI uses ruff)
uv pip install ruff
ruff check src/ tests/
```

## Required Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@localhost/bio_rag
SESSION_SECRET=<jwt-signing-key>
AI_INTEGRATIONS_OPENAI_API_KEY=<openai-api-key>
# Optional:
AI_INTEGRATIONS_OPENAI_BASE_URL=<custom-openai-base-url>  # For custom OpenAI endpoints
QDRANT_URL=<qdrant-cloud-url>  # Falls back to local ./qdrant_data
QDRANT_API_KEY=<qdrant-api-key>
```

## Architecture

### Layer Structure
```
main.py                    # FastAPI app entry, routes, lifespan
src/
├── api/                   # REST endpoints (auth, papers, chat)
│   ├── auth.py            # JWT auth with get_current_user/get_optional_user
│   ├── papers.py          # PubMed search, semantic search, indexing
│   ├── chat.py            # RAG query endpoint
│   └── schemas.py         # Pydantic request/response models
├── models/                # SQLAlchemy ORM (User, Paper, Chunk, ChatSession)
│   └── database.py        # Engine & session configuration
└── services/              # Business logic
    ├── rag_service.py     # Core RAG pipeline & reasoning mode
    ├── embedding_service.py  # PubMedBERT/OpenAI/Simple embeddings
    ├── vector_store.py    # Qdrant operations (singleton client)
    ├── pubmed_service.py  # PubMed E-utilities API
    └── translation_service.py  # Korean→English translation
static/                    # Frontend SPA (HTML + Tailwind + vanilla JS)
tests/
└── conftest.py            # TestableRAGService + mock fixtures
```

### Key Design Patterns

**Three-Tier Embedding Fallback**: The `EmbeddingService` class (`embedding_service.py:205`) tries models in order:
1. PubMedBERT (768d) - preferred for biomedical text, requires `transformers` + `torch`
2. OpenAI embeddings (1536d) - fallback, requires API key
3. SimpleEmbedding (768d) - hash-based fallback that works without external dependencies

Each embedding dimension uses a separate Qdrant collection: `biomedical_papers_{dimension}d`.

**RAG Pipeline Flow** (`rag_service.py`):
1. Query → (optional) translate Korean to English via `TranslationService`
2. Generate embedding → search Qdrant for similar chunks
3. Build context from top-K results → send to GPT-4o with citation prompt
4. Return `RAGResponse` with answer, sources, confidence score, and chunks_used

**Reasoning Mode** (`reasoning_query` method): For complex questions, decomposes into sub-questions via LLM, searches iteratively, then synthesizes a final answer with `reasoning_steps` for UI visualization.

**Bilingual Support**: `TranslationService` (`translation_service.py`) detects Korean (20%+ Korean characters via regex) and translates using GPT-4o with biomedical terminology awareness, falling back to Google Translate.

**Singleton Vector Store Client**: `_get_qdrant_client()` in `vector_store.py:18` ensures single Qdrant connection across the app.

### API Routes

- `POST /api/v1/auth/register|login` - JWT authentication (bcrypt + python-jose)
- `GET /api/v1/papers/search?query=` - PubMed E-utilities search
- `GET /api/v1/papers/semantic-search?query=` - Vector similarity search
- `POST /api/v1/papers/index` - Index single paper into vector DB (requires auth)
- `POST /api/v1/papers/index-from-pubmed?query=` - Search PubMed and index results (requires auth)
- `POST /api/v1/chat/query` - RAG query (supports `reasoning_mode` flag)
- `GET /api/v1/stats` - Get indexed chunk count
- `GET /api/health` - Health check

### Database Models

- **User**: UUID PK, email auth, timestamps
- **Paper**: PMID PK, title, abstract, full_text, journal, pub_date
- **Chunk**: Paper text chunks for RAG retrieval
- **ChatSession/ChatMessage**: User chat history with sources as JSON
- **Author/Keyword**: Many-to-many with papers

### Testing

Tests use a `_TestableRAGService` class in `conftest.py` that mirrors the real `RAGService` logic but accepts mocked dependencies. Key fixtures:
- `mock_embedding_service` - returns deterministic 768d vectors
- `mock_vector_store` - mocks Qdrant operations
- `mock_llm_client` - mocks OpenAI chat completions
- `sample_search_results` - predefined `MockSearchResult` objects

Async tests use `pytest-asyncio` with `asyncio_mode = "auto"`.
