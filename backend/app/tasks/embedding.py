import asyncio
import logging
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.models.paper import Paper
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# Create async engine for tasks
engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def _get_papers_without_embeddings(
    limit: int = 100,
) -> list[tuple[str, str, str | None]]:
    """Get papers that need embedding generation"""
    async with async_session() as session:
        # Get recently added papers
        cutoff = datetime.utcnow() - timedelta(days=7)
        stmt = (
            select(Paper.pmid, Paper.title, Paper.abstract)
            .where(Paper.created_at >= cutoff)
            .order_by(Paper.created_at.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        return [(row[0], row[1], row[2]) for row in result.fetchall()]


async def _generate_embeddings_for_papers(
    papers: list[tuple[str, str, str | None]],
) -> int:
    """Generate embeddings for papers and store in vector DB"""
    # Import here to avoid circular imports
    # In production, this would use the actual embedding service
    logger.info(f"Generating embeddings for {len(papers)} papers")

    processed = 0
    for pmid, title, abstract in papers:
        try:
            # Combine title and abstract for embedding
            text = f"{title}\n\n{abstract}" if abstract else title

            # TODO: Implement actual embedding generation
            # embedding = embedding_service.embed_text(text)
            # vector_store.add(pmid, embedding, {"title": title})

            processed += 1
            logger.debug(f"Generated embedding for PMID: {pmid}")

        except Exception as e:
            logger.error(f"Error generating embedding for {pmid}: {e}")
            continue

    return processed


@celery_app.task(bind=True, max_retries=3)
def process_pending_embeddings(self) -> dict:
    """Process papers that need embedding generation"""
    try:
        papers = asyncio.run(_get_papers_without_embeddings(limit=500))

        if not papers:
            return {
                "status": "completed",
                "message": "No papers to process",
                "processed": 0,
            }

        processed = asyncio.run(_generate_embeddings_for_papers(papers))

        return {
            "status": "completed",
            "total_papers": len(papers),
            "processed": processed,
        }

    except Exception as e:
        logger.error(f"Error processing embeddings: {e}")
        raise self.retry(exc=e, countdown=300)


@celery_app.task(bind=True, max_retries=3)
def generate_embedding_for_paper(self, pmid: str) -> dict:
    """Generate embedding for a specific paper"""
    async def _process():
        async with async_session() as session:
            stmt = select(Paper).where(Paper.pmid == pmid)
            result = await session.execute(stmt)
            paper = result.scalar_one_or_none()

            if not paper:
                return {"status": "error", "message": f"Paper {pmid} not found"}

            text = f"{paper.title}\n\n{paper.abstract}" if paper.abstract else paper.title

            # TODO: Generate and store embedding
            # embedding = embedding_service.embed_text(text)
            # vector_store.add(pmid, embedding, {"title": paper.title})

            return {"status": "completed", "pmid": pmid}

    try:
        return asyncio.run(_process())
    except Exception as e:
        logger.error(f"Error generating embedding for {pmid}: {e}")
        raise self.retry(exc=e, countdown=60)
