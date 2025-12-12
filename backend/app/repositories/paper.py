from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper import Paper, SavedPaper
from app.repositories.base import BaseRepository
from app.schemas.paper import PaperMetadata


class PaperRepository(BaseRepository[Paper]):
    """Paper repository"""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Paper, session)

    async def get_by_pmid(self, pmid: str) -> Paper | None:
        """Get paper by PMID"""
        stmt = select(Paper).where(Paper.pmid == pmid)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_pmids(self, pmids: list[str]) -> list[Paper]:
        """Get papers by PMID list"""
        stmt = select(Paper).where(Paper.pmid.in_(pmids))
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create_or_update(self, paper_data: PaperMetadata) -> Paper:
        """Create or update paper"""
        existing = await self.get_by_pmid(paper_data.pmid)

        if existing:
            for field, value in paper_data.model_dump().items():
                if value is not None:
                    setattr(existing, field, value)
            existing.updated_at = datetime.utcnow()
            await self.session.flush()
            await self.session.refresh(existing)
            return existing
        else:
            paper = Paper(**paper_data.model_dump())
            self.session.add(paper)
            await self.session.flush()
            await self.session.refresh(paper)
            return paper

    async def search_by_keyword(
        self,
        keyword: str,
        limit: int = 10,
        offset: int = 0,
    ) -> tuple[list[Paper], int]:
        """Search papers by keyword in title/abstract"""
        search_term = f"%{keyword}%"

        # Count query
        count_stmt = select(func.count()).where(
            (Paper.title.ilike(search_term)) | (Paper.abstract.ilike(search_term))
        )
        count_result = await self.session.execute(count_stmt)
        total = count_result.scalar() or 0

        # Data query
        stmt = (
            select(Paper)
            .where(
                (Paper.title.ilike(search_term)) | (Paper.abstract.ilike(search_term))
            )
            .order_by(Paper.publication_date.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        papers = list(result.scalars().all())

        return papers, total

    async def get_by_journal(
        self,
        journal: str,
        limit: int = 10,
    ) -> list[Paper]:
        """Get papers by journal"""
        stmt = (
            select(Paper)
            .where(Paper.journal.ilike(f"%{journal}%"))
            .order_by(Paper.publication_date.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_recent(self, limit: int = 10) -> list[Paper]:
        """Get recently added papers"""
        stmt = (
            select(Paper)
            .order_by(Paper.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def save_paper_for_user(
        self,
        user_id: UUID,
        pmid: str,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> SavedPaper:
        """Save paper for user"""
        saved = SavedPaper(
            user_id=user_id,
            pmid=pmid,
            tags=tags,
            notes=notes,
        )
        self.session.add(saved)
        await self.session.flush()
        await self.session.refresh(saved)
        return saved

    async def get_saved_papers(
        self,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[SavedPaper]:
        """Get user's saved papers"""
        stmt = (
            select(SavedPaper)
            .where(SavedPaper.user_id == user_id)
            .order_by(SavedPaper.saved_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def unsave_paper(self, user_id: UUID, pmid: str) -> bool:
        """Remove paper from user's saved list"""
        stmt = select(SavedPaper).where(
            (SavedPaper.user_id == user_id) & (SavedPaper.pmid == pmid)
        )
        result = await self.session.execute(stmt)
        saved = result.scalar_one_or_none()

        if saved:
            await self.session.delete(saved)
            await self.session.flush()
            return True
        return False
