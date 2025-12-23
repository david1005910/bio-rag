"""
Hybrid Search API - Dense + Sparse with RRF Fusion
Qdrant 기반 하이브리드 검색 API
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.pubmed.client import pubmed_client
from app.services.vector.qdrant_store import qdrant_store, HybridSearchResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hybrid", tags=["Hybrid Search"])


# ==================== Request/Response Models ====================

class IndexPapersRequest(BaseModel):
    """논문 인덱싱 요청"""
    query: str = Field(..., min_length=1, max_length=500, description="PubMed 검색 쿼리")
    max_papers: int = Field(default=50, ge=5, le=200, description="인덱싱할 최대 논문 수")


class IndexPapersResponse(BaseModel):
    """논문 인덱싱 결과"""
    indexed_count: int
    collection_name: str
    total_in_collection: int


class ScoreBreakdown(BaseModel):
    """점수 상세 분해"""
    dense_score: float = Field(..., description="Dense (semantic) 검색 점수")
    sparse_score: float = Field(..., description="Sparse (BM25) 검색 점수")
    rrf_score: float = Field(..., description="RRF 융합 점수")
    dense_rank: int = Field(..., description="Dense 검색 순위 (0=not found)")
    sparse_rank: int = Field(..., description="Sparse 검색 순위 (0=not found)")
    dense_contribution: float = Field(..., description="Dense 기여도 (%)")
    sparse_contribution: float = Field(..., description="Sparse 기여도 (%)")


class HybridSearchResultItem(BaseModel):
    """하이브리드 검색 결과 아이템"""
    doc_id: str
    pmid: str = ""
    title: str = ""
    content: str
    authors: list[str] = []
    journal: str = ""
    publication_date: str = ""
    scores: ScoreBreakdown


class ScoreDistribution(BaseModel):
    """점수 분포 통계"""
    min: float
    max: float
    avg: float


class ScoreVisualization(BaseModel):
    """RRF 점수 시각화 데이터"""
    dense_scores: ScoreDistribution
    sparse_scores: ScoreDistribution
    rrf_scores: ScoreDistribution
    contributions: list[dict[str, Any]]
    config: dict[str, float]


class HybridSearchResponse(BaseModel):
    """하이브리드 검색 응답"""
    query: str
    total_results: int
    dense_weight: float
    sparse_weight: float
    rrf_k: int
    results: list[HybridSearchResultItem]
    score_visualization: ScoreVisualization


# ==================== API Endpoints ====================

@router.post("/index", response_model=IndexPapersResponse)
async def index_papers_from_pubmed(request: IndexPapersRequest) -> IndexPapersResponse:
    """
    PubMed에서 논문을 검색하여 Qdrant에 인덱싱

    1. PubMed API로 논문 검색
    2. Dense vector (semantic embedding) 생성
    3. Sparse vector (BM25) 생성
    4. Qdrant에 저장
    """
    try:
        # Search PubMed
        pmids = await pubmed_client.search(request.query, max_results=request.max_papers)

        if not pmids:
            raise HTTPException(status_code=404, detail="No papers found")

        # Fetch paper details
        papers = await pubmed_client.fetch_papers(pmids)

        # Prepare documents for indexing
        documents = []
        for paper in papers:
            content = f"{paper.title}\n\n{paper.abstract or ''}"
            documents.append({
                "doc_id": f"pmid_{paper.pmid}",
                "content": content,
                "pmid": paper.pmid,
                "title": paper.title,
                "abstract": paper.abstract or "",
                "authors": paper.authors or [],
                "journal": paper.journal or "",
                "publication_date": paper.publication_date or "",
            })

        # Index to Qdrant
        indexed_count = qdrant_store.add_documents_batch(documents)

        return IndexPapersResponse(
            indexed_count=indexed_count,
            collection_name=qdrant_store.collection_name,
            total_in_collection=qdrant_store.count(),
        )

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.get("/search", response_model=HybridSearchResponse)
async def hybrid_search(
    query: str = Query(..., min_length=1, max_length=500, description="검색 쿼리"),
    top_k: int = Query(default=10, ge=1, le=50, description="반환할 결과 수"),
    dense_weight: float = Query(default=0.7, ge=0, le=1, description="Dense 검색 가중치"),
    sparse_weight: float = Query(default=0.3, ge=0, le=1, description="Sparse 검색 가중치"),
    rrf_k: int = Query(default=60, ge=1, le=100, description="RRF 상수 (높을수록 부드러운 융합)"),
) -> HybridSearchResponse:
    """
    Hybrid Search: Dense + Sparse with RRF Fusion

    **검색 방식:**
    - **Dense Search**: Semantic embedding으로 의미적 유사도 검색
    - **Sparse Search**: BM25 기반 키워드 매칭 검색
    - **RRF Fusion**: 두 결과를 Reciprocal Rank Fusion으로 통합

    **점수 설명:**
    - `dense_score`: 의미적 유사도 (0-1, 높을수록 유사)
    - `sparse_score`: 키워드 매칭 점수
    - `rrf_score`: 최종 융합 점수 (순위 결정에 사용)

    **가중치 조절:**
    - `dense_weight=1, sparse_weight=0`: 순수 의미 검색
    - `dense_weight=0, sparse_weight=1`: 순수 키워드 검색
    - `dense_weight=0.7, sparse_weight=0.3`: 의미 중심 하이브리드 (기본값)
    """
    try:
        # Check if collection has documents
        doc_count = qdrant_store.count()
        if doc_count == 0:
            raise HTTPException(
                status_code=404,
                detail="No documents indexed. Use POST /hybrid/index first."
            )

        # Perform hybrid search
        search_response = qdrant_store.hybrid_search(
            query=query,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rrf_k=rrf_k,
        )

        # Convert to response model
        results = []
        contributions = search_response.score_distribution.get("contributions", [])

        for i, result in enumerate(search_response.results):
            # Get contribution for this result
            contrib = contributions[i] if i < len(contributions) else {
                "dense_contribution": 50.0,
                "sparse_contribution": 50.0,
            }

            results.append(HybridSearchResultItem(
                doc_id=result.doc_id,
                pmid=result.metadata.get("pmid", ""),
                title=result.metadata.get("title", ""),
                content=result.content[:500] + "..." if len(result.content) > 500 else result.content,
                authors=result.metadata.get("authors", []),
                journal=result.metadata.get("journal", ""),
                publication_date=result.metadata.get("publication_date", ""),
                scores=ScoreBreakdown(
                    dense_score=round(result.dense_score, 4),
                    sparse_score=round(result.sparse_score, 4),
                    rrf_score=round(result.rrf_score, 6),
                    dense_rank=result.dense_rank,
                    sparse_rank=result.sparse_rank,
                    dense_contribution=contrib.get("dense_contribution", 50.0),
                    sparse_contribution=contrib.get("sparse_contribution", 50.0),
                ),
            ))

        # Build score visualization
        dist = search_response.score_distribution
        score_viz = ScoreVisualization(
            dense_scores=ScoreDistribution(**dist.get("dense_scores", {"min": 0, "max": 0, "avg": 0})),
            sparse_scores=ScoreDistribution(**dist.get("sparse_scores", {"min": 0, "max": 0, "avg": 0})),
            rrf_scores=ScoreDistribution(**dist.get("rrf_scores", {"min": 0, "max": 0, "avg": 0})),
            contributions=dist.get("contributions", []),
            config=dist.get("config", {"dense_weight": 0.7, "sparse_weight": 0.3}),
        )

        return HybridSearchResponse(
            query=query,
            total_results=search_response.total_results,
            dense_weight=search_response.dense_weight,
            sparse_weight=search_response.sparse_weight,
            rrf_k=search_response.rrf_k,
            results=results,
            score_visualization=score_viz,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/stats")
async def get_index_stats() -> dict[str, Any]:
    """
    인덱스 통계 조회
    """
    try:
        return {
            "collection_name": qdrant_store.collection_name,
            "document_count": qdrant_store.count(),
            "dense_dimension": qdrant_store.dense_dim,
            "config": {
                "dense_weight": qdrant_store.dense_weight,
                "sparse_weight": qdrant_store.sparse_weight,
                "rrf_k": qdrant_store.rrf_k,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_index() -> dict[str, str]:
    """
    인덱스 초기화 (모든 문서 삭제)
    """
    try:
        qdrant_store.clear()
        return {"message": "Index cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear index: {e}")
        raise HTTPException(status_code=500, detail=str(e))
