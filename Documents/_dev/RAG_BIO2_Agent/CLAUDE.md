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
```

## Required Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@localhost/bio_rag
SESSION_SECRET=<jwt-signing-key>
AI_INTEGRATIONS_OPENAI_API_KEY=<openai-api-key>
# Optional:
QDRANT_URL=<qdrant-cloud-url>  # Falls back to local ./qdrant_data
QDRANT_API_KEY=<qdrant-api-key>
```

## Architecture

### Layer Structure
```
main.py                    # FastAPI app entry, routes, lifespan
src/
├── api/                   # REST endpoints (auth, papers, chat)
│   └── schemas.py         # Pydantic request/response models
├── models/                # SQLAlchemy ORM (User, Paper, Chunk, ChatSession)
│   └── database.py        # Engine & session configuration
└── services/              # Business logic
    ├── rag_service.py     # Core RAG pipeline & reasoning mode
    ├── embedding_service.py  # PubMedBERT/OpenAI embeddings
    ├── vector_store.py    # Qdrant operations
    ├── pubmed_service.py  # PubMed E-utilities API
    └── translation_service.py  # Korean→English translation
static/                    # Frontend SPA (HTML + Tailwind + vanilla JS)
```

### Key Design Patterns

**Dual Embedding Models**: PubMedBERT (768d) is preferred for biomedical text; OpenAI embeddings (1536d) is the fallback. Each uses a separate Qdrant collection: `biomedical_papers_768d` or `biomedical_papers_1536d`.

**RAG Pipeline Flow**:
1. Query → (optional) translate Korean to English
2. Generate embedding → search Qdrant for similar chunks
3. Build context from top-K results → send to GPT-4o with citation prompt
4. Return answer with sources and confidence score

**Reasoning Mode**: For complex questions, decomposes into sub-questions, searches iteratively, then synthesizes a final answer with `reasoning_steps` for UI visualization.

**Bilingual Support**: Automatic Korean detection (20%+ Korean characters) triggers GPT-4o translation before search operations.

### API Routes

- `POST /api/v1/auth/register|login` - JWT authentication
- `GET /api/v1/papers/search?query=` - PubMed search
- `GET /api/v1/papers/semantic-search?query=` - Vector similarity search
- `POST /api/v1/papers/index` - Index paper into vector DB
- `POST /api/v1/chat/query` - RAG query (supports `reasoning_mode` flag)
- `GET /api/health` - Health check

### Database Models

- **User**: UUID PK, email auth, timestamps
- **Paper**: PMID PK, title, abstract, full_text, journal, pub_date
- **Chunk**: Paper text chunks for RAG retrieval
- **ChatSession/ChatMessage**: User chat history with sources as JSON
- **Author/Keyword**: Many-to-many with papers
