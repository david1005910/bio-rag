from celery import Celery
from celery.schedules import crontab

from app.core.config import settings

celery_app = Celery(
    "bio-rag",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.tasks.crawler",
        "app.tasks.embedding",
    ],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    "daily-paper-crawl": {
        "task": "app.tasks.crawler.crawl_daily_papers",
        "schedule": crontab(hour=2, minute=0),  # 02:00 UTC daily
        "options": {"queue": "crawler"},
    },
    "generate-embeddings": {
        "task": "app.tasks.embedding.process_pending_embeddings",
        "schedule": crontab(hour=3, minute=0),  # 03:00 UTC daily
        "options": {"queue": "embedding"},
    },
}

# Task routing
celery_app.conf.task_routes = {
    "app.tasks.crawler.*": {"queue": "crawler"},
    "app.tasks.embedding.*": {"queue": "embedding"},
}
