import asyncio
import logging
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.models.paper import Paper
from app.services.pubmed.client import pubmed_client
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# Create async engine for tasks
engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def _crawl_papers(
    query: str,
    max_results: int = 1000,
    days_back: int = 7,
) -> int:
    """Crawl papers from PubMed and save to database"""
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    date_range = (
        start_date.strftime("%Y/%m/%d"),
        end_date.strftime("%Y/%m/%d"),
    )

    logger.info(f"Crawling papers: query='{query}', date_range={date_range}")

    # Search for PMIDs
    pmids = await pubmed_client.search(
        query=query,
        max_results=max_results,
        date_range=date_range,
    )

    if not pmids:
        logger.info("No papers found")
        return 0

    logger.info(f"Found {len(pmids)} papers")

    # Fetch paper metadata
    papers = await pubmed_client.batch_fetch(pmids)

    # Save to database
    saved_count = 0
    async with async_session() as session:
        for paper_data in papers:
            # Check if already exists
            stmt = select(Paper).where(Paper.pmid == paper_data.pmid)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing paper
                for field, value in paper_data.model_dump().items():
                    if value is not None:
                        setattr(existing, field, value)
                existing.updated_at = datetime.utcnow()
            else:
                # Create new paper
                new_paper = Paper(**paper_data.model_dump())
                session.add(new_paper)
                saved_count += 1

        await session.commit()

    logger.info(f"Saved {saved_count} new papers")
    return saved_count


@celery_app.task(bind=True, max_retries=3)
def crawl_daily_papers(self) -> dict:
    """Daily paper crawling task"""
    queries = [
        "cancer immunotherapy",
        "CRISPR gene editing",
        "machine learning drug discovery",
        "single cell RNA sequencing",
        "protein structure prediction",
    ]

    total_saved = 0
    results = {}

    for query in queries:
        try:
            saved = asyncio.run(
                _crawl_papers(query=query, max_results=200, days_back=1)
            )
            results[query] = saved
            total_saved += saved
        except Exception as e:
            logger.error(f"Error crawling '{query}': {e}")
            results[query] = f"error: {e}"

    return {
        "status": "completed",
        "total_saved": total_saved,
        "details": results,
    }


@celery_app.task(bind=True, max_retries=3)
def crawl_papers_by_query(
    self,
    query: str,
    max_results: int = 100,
    days_back: int = 30,
) -> dict:
    """Crawl papers for a specific query"""
    try:
        saved = asyncio.run(
            _crawl_papers(query=query, max_results=max_results, days_back=days_back)
        )
        return {
            "status": "completed",
            "query": query,
            "saved_count": saved,
        }
    except Exception as e:
        logger.error(f"Error crawling '{query}': {e}")
        raise self.retry(exc=e, countdown=60)
