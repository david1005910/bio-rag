from typing import Any, Generic, TypeVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository with common CRUD operations"""

    def __init__(self, model: type[ModelType], session: AsyncSession) -> None:
        self.model = model
        self.session = session

    async def get_by_id(self, id: UUID | str) -> ModelType | None:
        """Get entity by ID"""
        id_value = str(id) if isinstance(id, UUID) else id
        stmt = select(self.model).where(
            getattr(self.model, self._get_pk_name()) == id_value
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelType]:
        """Get all entities with pagination"""
        stmt = select(self.model).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, data: dict[str, Any]) -> ModelType:
        """Create new entity"""
        entity = self.model(**data)
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def update(
        self,
        id: UUID | str,
        data: dict[str, Any],
    ) -> ModelType | None:
        """Update entity"""
        id_value = str(id) if isinstance(id, UUID) else id
        entity = await self.get_by_id(id_value)
        if entity is None:
            return None

        for field, value in data.items():
            if hasattr(entity, field) and value is not None:
                setattr(entity, field, value)

        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def delete(self, id: UUID | str) -> bool:
        """Delete entity"""
        entity = await self.get_by_id(id)
        if entity is None:
            return False

        await self.session.delete(entity)
        await self.session.flush()
        return True

    async def count(self) -> int:
        """Count all entities"""
        from sqlalchemy import func

        stmt = select(func.count()).select_from(self.model)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    def _get_pk_name(self) -> str:
        """Get primary key column name"""
        pk_columns = self.model.__table__.primary_key.columns
        return pk_columns.keys()[0]
