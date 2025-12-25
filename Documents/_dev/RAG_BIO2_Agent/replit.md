# Bio-RAG Platform

## Overview

Bio-RAG (Biomedical Research AI-Guided Analytics) is an AI-powered biomedical research paper analysis platform. It uses RAG (Retrieval-Augmented Generation) technology to help researchers search, analyze, and extract insights from scientific literature. The platform integrates with PubMed for paper retrieval, uses vector embeddings for semantic search, and provides a conversational AI interface for querying research papers.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Framework
- **FastAPI** serves as the main API framework with async support
- RESTful API design with versioned endpoints (`/api/v1/`)
- Lifespan context manager for database initialization on startup

### Authentication System
- JWT-based authentication using `python-jose`
- Password hashing with bcrypt via `passlib`
- Bearer token authentication with optional user endpoints for public access

### Database Layer
- **PostgreSQL** with SQLAlchemy ORM (uses `DATABASE_URL` environment variable)
- Models include: Users, Papers, Authors, Keywords, Chunks, ChatSessions, ChatMessages
- Many-to-many relationships for paper-author and paper-keyword associations
- UUID primary keys for user-related tables, PMID strings for papers

### Vector Storage & Embeddings
- **Qdrant** vector database for semantic search (supports both cloud and local storage)
- Dual embedding model support:
  - **PubMedBERT** (768 dimensions) - preferred for biomedical domain
  - **OpenAI embeddings** (1536 dimensions) - fallback option
- Collection naming convention: `biomedical_papers_{dimension}d`

### RAG Pipeline
- Embedding service abstracts model selection
- Vector store handles similarity search with metadata filtering
- RAG service combines retrieval with OpenAI LLM for answer generation
- System prompt enforces citation requirements and factual responses

### External Data Integration
- **PubMed E-utilities API** for paper search and metadata fetching
- Retry logic with exponential backoff using `tenacity`
- Rate limiting based on API key availability

### Frontend
- Static HTML/CSS/JS served from `/static` directory
- Tailwind CSS for styling
- Vanilla JavaScript for API interactions
- Single-page application with modal-based authentication

## External Dependencies

### Required Environment Variables
- `DATABASE_URL` - PostgreSQL connection string
- `SESSION_SECRET` - JWT signing key
- `AI_INTEGRATIONS_OPENAI_BASE_URL` - OpenAI API base URL
- `AI_INTEGRATIONS_OPENAI_API_KEY` - OpenAI API key

### Optional Environment Variables
- `QDRANT_URL` - Qdrant cloud URL (falls back to local storage)
- `QDRANT_API_KEY` - Qdrant cloud API key

## Recent Changes (December 2024)
- **Added bilingual search (Korean/English)**: Korean queries are automatically translated to English for PubMed API and semantic search
  - TranslationService uses GPT-4o for accurate biomedical terminology translation
  - Works in paper search, semantic search, and chat queries
- **Added Reasoning RAG mode**: Multi-step reasoning for complex questions
  - Question decomposition into sub-questions
  - Iterative search and analysis for each sub-question
  - Answer synthesis with chain-of-thought reasoning
  - Visual display of reasoning steps in the UI
- Fixed embedding dimension mismatch issue: PubMedBERT (768d) and OpenAI (1536d) now use separate Qdrant collections
- Collection naming: `biomedical_papers_768d` for PubMedBERT, `biomedical_papers_1536d` for OpenAI
- Added email-validator package for Pydantic EmailStr validation
- Fixed Integer import in user.py model

### Key Python Dependencies
- `fastapi` - Web framework
- `sqlalchemy` - Database ORM
- `qdrant-client` - Vector database client
- `transformers` + `torch` - PubMedBERT embeddings
- `openai` - LLM integration
- `httpx` - Async HTTP client for PubMed API
- `python-jose` - JWT handling
- `passlib` - Password hashing

### External Services
- **PubMed/NCBI E-utilities** - Paper metadata and search
- **OpenAI API** - LLM for RAG responses
- **Qdrant** - Vector similarity search (cloud or local)