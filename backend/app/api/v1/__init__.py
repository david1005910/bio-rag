"""API v1 router"""

from fastapi import APIRouter

from app.api.v1.analytics import router as analytics_router
from app.api.v1.arxiv import router as arxiv_router
from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router
from app.api.v1.documents import router as documents_router
from app.api.v1.hybrid_search import router as hybrid_router
from app.api.v1.i18n import router as i18n_router
from app.api.v1.pubmed import router as pubmed_router
from app.api.v1.search import router as search_router
from app.api.v1.users import router as users_router

api_router = APIRouter()

# Core endpoints
api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(search_router)
api_router.include_router(chat_router)

# New service endpoints
api_router.include_router(arxiv_router)
api_router.include_router(pubmed_router)
api_router.include_router(analytics_router)
api_router.include_router(documents_router)
api_router.include_router(i18n_router)

# Hybrid Search (Qdrant + PubMedBERT + SPLADE)
api_router.include_router(hybrid_router)
