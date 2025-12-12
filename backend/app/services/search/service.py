"""Search service for paper retrieval"""

import logging
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import SearchLog
from app.repositories.paper import PaperRepository
from app.schemas.paper import PaperSummary
from app.schemas.search import SearchFilters, SearchResult
from app.services.rag.hybrid_search import HybridSearcher, hybrid_searcher
from app.services.rag.reranker import CrossEncoderReranker, cross_encoder_reranker
from app.services.rag.retriever import RetrievedDocument

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Search configuration"""

    use_hybrid: bool = True
    use_reranking: bool = True
    initial_candidates: int = 50
    final_results: int = 10


class SearchService:
    """Service for searching papers"""

    def __init__(
        self,
        db: AsyncSession,
        hybrid_searcher: HybridSearcher | None = None,
        reranker: CrossEncoderReranker | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        self.db = db
        self.paper_repo = PaperRepository(db)
        self._hybrid_searcher = hybrid_searcher
        self._reranker = reranker
        self.config = config or SearchConfig()

    @property
    def searcher(self) -> HybridSearcher:
        """Get hybrid searcher"""
        if self._hybrid_searcher is None:
            self._hybrid_searcher = hybrid_searcher
        return self._hybrid_searcher

    @property
    def reranker(self) -> CrossEncoderReranker:
        """Get reranker"""
        if self._reranker is None:
            self._reranker = cross_encoder_reranker
        return self._reranker

    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 10,
        offset: int = 0,
        user_id: UUID | None = None,
    ) -> SearchResult:
        """
        Search for papers

        Args:
            query: Search query
            filters: Optional search filters
            limit: Number of results
            offset: Pagination offset
            user_id: Optional user ID for logging

        Returns:
            SearchResult with matching papers
        """
        start_time = time.time()

        # Build metadata filter from SearchFilters
        filter_metadata = self._build_filter_metadata(filters)

        # Get more candidates for reranking
        candidate_count = limit + offset + self.config.initial_candidates

        # Search using hybrid search
        if self.config.use_hybrid:
            documents = self.searcher.search(
                query=query,
                top_k=candidate_count,
                filter_metadata=filter_metadata,
            )
        else:
            documents = self.searcher.dense_retriever.retrieve(
                query=query,
                top_k=candidate_count,
                filter_metadata=filter_metadata,
            )

        # Apply reranking
        if self.config.use_reranking and documents:
            documents = self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=candidate_count,
            )

        # Apply pagination
        paginated_docs = documents[offset : offset + limit]

        # Convert to PaperSummary
        results = await self._documents_to_summaries(paginated_docs)

        query_time_ms = int((time.time() - start_time) * 1000)

        # Log search if user is authenticated
        if user_id:
            await self._log_search(user_id, query, len(results), query_time_ms)

        return SearchResult(
            results=results,
            total=len(documents),
            query_time_ms=query_time_ms,
        )

    async def get_similar_papers(
        self,
        pmid: str,
        limit: int = 5,
    ) -> list[PaperSummary]:
        """
        Get papers similar to a given paper

        Args:
            pmid: Paper PMID
            limit: Number of similar papers

        Returns:
            List of similar papers
        """
        # Get paper from database
        paper = await self.paper_repo.get_by_pmid(pmid)
        if not paper:
            return []

        # Use title + abstract as query
        query = f"{paper.title} {paper.abstract or ''}"

        # Search for similar papers
        documents = self.searcher.search(
            query=query,
            top_k=limit + 1,  # +1 to exclude the source paper
        )

        # Filter out the source paper
        documents = [doc for doc in documents if doc.pmid != pmid][:limit]

        return await self._documents_to_summaries(documents)

    async def search_by_pmids(
        self,
        pmids: list[str],
    ) -> list[PaperSummary]:
        """
        Get papers by PMIDs

        Args:
            pmids: List of PMIDs

        Returns:
            List of paper summaries
        """
        summaries: list[PaperSummary] = []

        for pmid in pmids:
            paper = await self.paper_repo.get_by_pmid(pmid)
            if paper:
                summaries.append(
                    PaperSummary(
                        pmid=paper.pmid,
                        title=paper.title,
                        authors=paper.authors[:3] if paper.authors else [],
                        journal=paper.journal or "",
                        pub_date=paper.publication_date,
                        abstract_snippet=paper.abstract[:300] if paper.abstract else None,
                    )
                )

        return summaries

    def _build_filter_metadata(
        self,
        filters: SearchFilters | None,
    ) -> dict[str, Any] | None:
        """Build metadata filter from SearchFilters"""
        if not filters:
            return None

        filter_dict: dict[str, Any] = {}

        if filters.year_from:
            filter_dict["year_from"] = filters.year_from
        if filters.year_to:
            filter_dict["year_to"] = filters.year_to
        if filters.journals:
            filter_dict["journals"] = filters.journals
        if filters.authors:
            filter_dict["authors"] = filters.authors

        return filter_dict if filter_dict else None

    async def _documents_to_summaries(
        self,
        documents: list[RetrievedDocument],
    ) -> list[PaperSummary]:
        """Convert RetrievedDocuments to PaperSummaries"""
        summaries: list[PaperSummary] = []
        seen_pmids: set[str] = set()

        for doc in documents:
            if doc.pmid in seen_pmids:
                continue
            seen_pmids.add(doc.pmid)

            # Try to get full paper info from database
            paper = await self.paper_repo.get_by_pmid(doc.pmid)

            if paper:
                summaries.append(
                    PaperSummary(
                        pmid=paper.pmid,
                        title=paper.title,
                        authors=paper.authors[:3] if paper.authors else [],
                        journal=paper.journal or "",
                        pub_date=paper.publication_date,
                        abstract_snippet=paper.abstract[:300] if paper.abstract else None,
                    )
                )
            else:
                # Use metadata from document
                summaries.append(
                    PaperSummary(
                        pmid=doc.pmid,
                        title=doc.title,
                        authors=doc.metadata.get("authors", [])[:3],
                        journal=doc.metadata.get("journal", ""),
                        pub_date=doc.metadata.get("pub_date"),
                        abstract_snippet=doc.content[:300] if doc.content else None,
                    )
                )

        return summaries

    async def _log_search(
        self,
        user_id: UUID,
        query: str,
        result_count: int,
        latency_ms: int,
    ) -> None:
        """Log search query"""
        try:
            search_log = SearchLog(
                user_id=user_id,
                query=query,
                result_count=result_count,
                latency_ms=latency_ms,
            )
            self.db.add(search_log)
            await self.db.commit()
        except Exception as e:
            logger.warning(f"Failed to log search: {e}")
            await self.db.rollback()
