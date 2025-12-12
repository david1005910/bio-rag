# Bio-RAG

**Biomedical Research AI-Guided Analytics Platform**

RAG 기반 바이오의학 논문 검색 및 AI Q&A 플랫폼

## Features

- **Semantic Search**: PubMedBERT 임베딩 기반 의미 검색
- **Hybrid Search**: Dense + BM25 검색과 RRF fusion
- **Cross-Encoder Re-ranking**: 검색 결과 재순위화
- **AI Q&A**: GPT-4 기반 논문 질의응답 (출처 인용 포함)
- **Citation Verification**: 출처 검증 및 할루시네이션 탐지
- **Streaming Response**: 실시간 스트리밍 응답
- **Paper Recommendations**: 유사 논문 추천

## Tech Stack

### Backend
- **Framework**: Python 3.11+, FastAPI, Pydantic v2
- **Database**: PostgreSQL (AsyncPG), Redis
- **Vector Store**: ChromaDB
- **Task Queue**: Celery with Redis
- **AI/ML**: LangChain, OpenAI GPT-4, PubMedBERT
- **Authentication**: JWT (python-jose)

### Frontend
- **Framework**: TypeScript, React 18
- **Styling**: TailwindCSS
- **State Management**: React Query (TanStack Query)
- **Routing**: React Router v6

### Infrastructure
- **Cloud**: AWS (EKS, RDS, ElastiCache, S3, ECR)
- **IaC**: Terraform
- **Container**: Docker, Kubernetes
- **CI/CD**: GitHub Actions

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Node.js 20+
- Git

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/bio-rag.git
cd bio-rag

# Start infrastructure services
docker-compose up -d postgres redis chromadb

# Backend setup
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Run database migrations
alembic upgrade head

# Start backend server
uvicorn app.main:app --reload --port 8000

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev
```

### Using Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## Project Structure

```
bio-rag/
├── backend/
│   ├── app/
│   │   ├── api/              # API endpoints
│   │   │   └── v1/           # API v1 routes
│   │   ├── core/             # Core configuration
│   │   ├── db/               # Database setup
│   │   ├── models/           # SQLAlchemy models
│   │   ├── repositories/     # Data access layer
│   │   ├── schemas/          # Pydantic schemas
│   │   ├── services/         # Business logic
│   │   │   ├── auth/         # Authentication
│   │   │   ├── chat/         # Chat service
│   │   │   ├── pubmed/       # PubMed API client
│   │   │   ├── rag/          # RAG components
│   │   │   ├── search/       # Search service
│   │   │   └── vector/       # Vector store
│   │   └── tasks/            # Celery tasks
│   ├── alembic/              # Database migrations
│   └── tests/                # Test files
├── frontend/
│   └── src/
│       ├── api/              # API client
│       ├── components/       # React components
│       ├── contexts/         # React contexts
│       ├── hooks/            # Custom hooks
│       ├── pages/            # Page components
│       └── types/            # TypeScript types
├── ml/
│   ├── chunking/             # Text chunking
│   ├── embeddings/           # Embedding models
│   └── pipeline/             # ML pipelines
├── infra/
│   ├── terraform/            # Infrastructure as Code
│   ├── kubernetes/           # K8s manifests
│   └── scripts/              # Deployment scripts
└── docker-compose.yml
```

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh tokens

### Users
- `GET /api/v1/users/me` - Get current user
- `PATCH /api/v1/users/me` - Update user
- `DELETE /api/v1/users/me` - Delete user

### Search
- `POST /api/v1/search` - Search papers
- `GET /api/v1/search/similar/{pmid}` - Get similar papers
- `POST /api/v1/search/by-pmids` - Get papers by PMIDs

### Chat
- `POST /api/v1/chat/sessions` - Create chat session
- `GET /api/v1/chat/sessions` - List sessions
- `GET /api/v1/chat/sessions/{id}` - Get session
- `DELETE /api/v1/chat/sessions/{id}` - Delete session
- `POST /api/v1/chat/query` - Ask question
- `POST /api/v1/chat/query/stream` - Ask with streaming

## Configuration

### Environment Variables

```bash
# App
APP_NAME=Bio-RAG
DEBUG=false

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/biorag

# Redis
REDIS_URL=redis://localhost:6379/0

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8000

# OpenAI
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4-turbo-preview

# PubMed
PUBMED_API_KEY=your-api-key

# JWT
JWT_SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7
```

## Testing

```bash
# Backend tests
cd backend
pytest

# With coverage
pytest --cov=app --cov-report=html

# Frontend tests
cd frontend
npm run test
```

## Deployment

### Deploy to AWS

```bash
# Initialize Terraform
cd infra/terraform
terraform init

# Plan deployment
terraform plan -var-file=environments/prod.tfvars

# Apply infrastructure
terraform apply -var-file=environments/prod.tfvars

# Deploy application
cd ../..
./infra/scripts/deploy.sh prod
```

### Manual Kubernetes Deployment

```bash
# Configure kubectl
aws eks update-kubeconfig --region ap-northeast-2 --name bio-rag-prod

# Apply manifests
kubectl apply -k infra/kubernetes/
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   FastAPI   │────▶│  PostgreSQL │
│   (React)   │     │   Backend   │     │             │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │  ChromaDB │   │   Redis   │   │  Celery   │
    │  (Vector) │   │  (Cache)  │   │ (Workers) │
    └───────────┘   └───────────┘   └───────────┘
```

## RAG Pipeline

1. **Query Processing**: 사용자 질의 분석 및 언어 감지
2. **Retrieval**: Hybrid Search (Dense + BM25) + RRF Fusion
3. **Re-ranking**: Cross-Encoder 기반 재순위화
4. **Generation**: GPT-4 기반 답변 생성
5. **Validation**: Citation 검증 및 Hallucination 탐지

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.
