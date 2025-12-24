"""Tests for repository layer"""

from datetime import datetime
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatMessage, ChatSession
from app.models.paper import Paper, SavedPaper
from app.models.user import User
from app.repositories.base import BaseRepository
from app.repositories.chat import ChatRepository
from app.repositories.paper import PaperRepository
from app.repositories.user import UserRepository
from app.schemas.paper import PaperMetadata
from app.services.auth.security import hash_password


class TestBaseRepository:
    """Tests for BaseRepository"""

    @pytest_asyncio.fixture
    async def user_repo(self, db_session: AsyncSession):
        """Create user repository instance"""
        return BaseRepository(User, db_session)

    @pytest.mark.asyncio
    async def test_create(self, db_session: AsyncSession):
        """Test creating an entity"""
        repo = BaseRepository(User, db_session)
        user_id = str(uuid4())
        user = await repo.create({
            "user_id": user_id,
            "email": f"test_{uuid4()}@example.com",
            "password_hash": hash_password("password123"),
            "name": "Test User",
            "subscription_tier": "free",
            "is_active": True,
        })
        assert user is not None
        assert user.email.endswith("@example.com")

    @pytest.mark.asyncio
    async def test_get_by_id(self, db_session: AsyncSession, test_user: User):
        """Test getting entity by ID"""
        repo = BaseRepository(User, db_session)
        found = await repo.get_by_id(test_user.user_id)
        assert found is not None
        assert found.email == test_user.email

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, db_session: AsyncSession):
        """Test getting non-existent entity"""
        repo = BaseRepository(User, db_session)
        found = await repo.get_by_id(str(uuid4()))
        assert found is None

    @pytest.mark.asyncio
    async def test_get_all(self, db_session: AsyncSession, test_user: User):
        """Test getting all entities"""
        repo = BaseRepository(User, db_session)
        users = await repo.get_all(limit=10, offset=0)
        assert len(users) >= 1

    @pytest.mark.asyncio
    async def test_update(self, db_session: AsyncSession, test_user: User):
        """Test updating an entity"""
        repo = BaseRepository(User, db_session)
        updated = await repo.update(test_user.user_id, {"name": "Updated Name"})
        assert updated is not None
        assert updated.name == "Updated Name"

    @pytest.mark.asyncio
    async def test_update_not_found(self, db_session: AsyncSession):
        """Test updating non-existent entity"""
        repo = BaseRepository(User, db_session)
        updated = await repo.update(str(uuid4()), {"name": "Updated"})
        assert updated is None

    @pytest.mark.asyncio
    async def test_delete(self, db_session: AsyncSession):
        """Test deleting an entity"""
        repo = BaseRepository(User, db_session)
        user = await repo.create({
            "user_id": str(uuid4()),
            "email": f"delete_test_{uuid4()}@example.com",
            "password_hash": hash_password("password123"),
            "name": "Delete Test",
            "subscription_tier": "free",
            "is_active": True,
        })
        result = await repo.delete(user.user_id)
        assert result is True
        found = await repo.get_by_id(user.user_id)
        assert found is None

    @pytest.mark.asyncio
    async def test_delete_not_found(self, db_session: AsyncSession):
        """Test deleting non-existent entity"""
        repo = BaseRepository(User, db_session)
        result = await repo.delete(str(uuid4()))
        assert result is False

    @pytest.mark.asyncio
    async def test_count(self, db_session: AsyncSession, test_user: User):
        """Test counting entities"""
        repo = BaseRepository(User, db_session)
        count = await repo.count()
        assert count >= 1


class TestPaperRepository:
    """Tests for PaperRepository"""

    @pytest_asyncio.fixture
    async def paper_repo(self, db_session: AsyncSession):
        """Create paper repository instance"""
        return PaperRepository(db_session)

    @pytest_asyncio.fixture
    async def test_paper(self, db_session: AsyncSession) -> Paper:
        """Create test paper"""
        paper = Paper(
            pmid="TEST12345",
            title="Test Paper for Repository Tests",
            abstract="This is a test abstract for testing repository methods.",
            authors=["Author A", "Author B"],
            journal="Test Journal",
            publication_date=datetime(2024, 1, 15),
            keywords=["test", "repository"],
        )
        db_session.add(paper)
        await db_session.commit()
        await db_session.refresh(paper)
        return paper

    @pytest.mark.asyncio
    async def test_get_by_pmid(self, paper_repo: PaperRepository, test_paper: Paper):
        """Test getting paper by PMID"""
        paper = await paper_repo.get_by_pmid(test_paper.pmid)
        assert paper is not None
        assert paper.title == test_paper.title

    @pytest.mark.asyncio
    async def test_get_by_pmid_not_found(self, paper_repo: PaperRepository):
        """Test getting non-existent paper"""
        paper = await paper_repo.get_by_pmid("NONEXISTENT")
        assert paper is None

    @pytest.mark.asyncio
    async def test_get_by_pmids(self, paper_repo: PaperRepository, test_paper: Paper):
        """Test getting papers by PMID list"""
        papers = await paper_repo.get_by_pmids([test_paper.pmid, "NONEXISTENT"])
        assert len(papers) == 1
        assert papers[0].pmid == test_paper.pmid

    @pytest.mark.asyncio
    async def test_create_or_update_create(self, paper_repo: PaperRepository):
        """Test creating new paper"""
        paper_data = PaperMetadata(
            pmid=f"NEW{uuid4().hex[:8]}",
            title="New Test Paper",
            abstract="Abstract for new paper",
            authors=["New Author"],
            journal="New Journal",
            publication_date=datetime(2024, 6, 1),
        )
        paper = await paper_repo.create_or_update(paper_data)
        assert paper is not None
        assert paper.title == "New Test Paper"

    @pytest.mark.asyncio
    async def test_create_or_update_update(
        self, paper_repo: PaperRepository, test_paper: Paper
    ):
        """Test updating existing paper"""
        paper_data = PaperMetadata(
            pmid=test_paper.pmid,
            title="Updated Title",
            abstract=test_paper.abstract,
            authors=test_paper.authors,
            journal=test_paper.journal,
            publication_date=test_paper.publication_date,
        )
        paper = await paper_repo.create_or_update(paper_data)
        assert paper.title == "Updated Title"

    @pytest.mark.asyncio
    async def test_search_by_keyword(
        self, paper_repo: PaperRepository, test_paper: Paper
    ):
        """Test searching papers by keyword"""
        papers, total = await paper_repo.search_by_keyword("repository", limit=10)
        assert total >= 1
        assert any(p.pmid == test_paper.pmid for p in papers)

    @pytest.mark.asyncio
    async def test_search_by_keyword_no_results(self, paper_repo: PaperRepository):
        """Test searching with no results"""
        papers, total = await paper_repo.search_by_keyword("xyznonexistent123")
        assert total == 0
        assert len(papers) == 0

    @pytest.mark.asyncio
    async def test_get_by_journal(
        self, paper_repo: PaperRepository, test_paper: Paper
    ):
        """Test getting papers by journal"""
        papers = await paper_repo.get_by_journal("Test Journal")
        assert len(papers) >= 1

    @pytest.mark.asyncio
    async def test_get_recent(self, paper_repo: PaperRepository, test_paper: Paper):
        """Test getting recent papers"""
        papers = await paper_repo.get_recent(limit=5)
        assert len(papers) >= 1

    @pytest.mark.asyncio
    async def test_save_paper_for_user(
        self,
        paper_repo: PaperRepository,
        test_user: User,
        test_paper: Paper,
    ):
        """Test saving paper for user"""
        saved = await paper_repo.save_paper_for_user(
            user_id=test_user.user_id,
            pmid=test_paper.pmid,
            tags=["important", "review"],
            notes="Test notes",
        )
        assert saved is not None
        assert saved.pmid == test_paper.pmid

    @pytest.mark.asyncio
    async def test_get_saved_papers(
        self,
        db_session: AsyncSession,
        paper_repo: PaperRepository,
        test_user: User,
        test_paper: Paper,
    ):
        """Test getting user's saved papers"""
        # First save a paper
        await paper_repo.save_paper_for_user(
            user_id=test_user.user_id,
            pmid=test_paper.pmid,
        )
        await db_session.commit()

        saved_papers = await paper_repo.get_saved_papers(test_user.user_id)
        assert len(saved_papers) >= 1

    @pytest.mark.asyncio
    async def test_unsave_paper(
        self,
        db_session: AsyncSession,
        paper_repo: PaperRepository,
        test_user: User,
        test_paper: Paper,
    ):
        """Test removing paper from saved list"""
        # First save a paper
        await paper_repo.save_paper_for_user(
            user_id=test_user.user_id,
            pmid=test_paper.pmid,
        )
        await db_session.commit()

        result = await paper_repo.unsave_paper(test_user.user_id, test_paper.pmid)
        assert result is True

    @pytest.mark.asyncio
    async def test_unsave_paper_not_found(
        self, paper_repo: PaperRepository, test_user: User
    ):
        """Test unsaving non-existent saved paper"""
        result = await paper_repo.unsave_paper(test_user.user_id, "NONEXISTENT")
        assert result is False


class TestChatRepository:
    """Tests for ChatRepository"""

    @pytest_asyncio.fixture
    async def chat_repo(self, db_session: AsyncSession):
        """Create chat repository instance"""
        return ChatRepository(db_session)

    @pytest_asyncio.fixture
    async def test_session(
        self, db_session: AsyncSession, test_user: User
    ) -> ChatSession:
        """Create test chat session"""
        session = ChatSession(
            user_id=test_user.user_id,
            title="Test Chat Session",
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        return session

    @pytest.mark.asyncio
    async def test_create_session(
        self, chat_repo: ChatRepository, test_user: User
    ):
        """Test creating chat session"""
        session = await chat_repo.create_session(
            user_id=test_user.user_id,
            title="New Session",
        )
        assert session is not None
        assert session.title == "New Session"

    @pytest.mark.asyncio
    async def test_get_session(
        self, chat_repo: ChatRepository, test_session: ChatSession
    ):
        """Test getting chat session"""
        session = await chat_repo.get_session(test_session.session_id)
        assert session is not None
        assert session.title == test_session.title

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, chat_repo: ChatRepository):
        """Test getting non-existent session"""
        session = await chat_repo.get_session(uuid4())
        assert session is None

    @pytest.mark.asyncio
    async def test_get_session_with_messages(
        self,
        db_session: AsyncSession,
        chat_repo: ChatRepository,
        test_session: ChatSession,
    ):
        """Test getting session with messages loaded"""
        # Add a message first
        await chat_repo.add_message(
            session_id=test_session.session_id,
            role="user",
            content="Test message",
        )
        await db_session.commit()

        session = await chat_repo.get_session_with_messages(test_session.session_id)
        assert session is not None
        assert len(session.messages) >= 1

    @pytest.mark.asyncio
    async def test_list_sessions(
        self, chat_repo: ChatRepository, test_user: User, test_session: ChatSession
    ):
        """Test listing user sessions"""
        sessions, total = await chat_repo.list_sessions(test_user.user_id)
        assert total >= 1
        assert len(sessions) >= 1

    @pytest.mark.asyncio
    async def test_update_session_title(
        self, chat_repo: ChatRepository, test_session: ChatSession
    ):
        """Test updating session title"""
        updated = await chat_repo.update_session_title(
            test_session.session_id, "Updated Title"
        )
        assert updated is not None
        assert updated.title == "Updated Title"

    @pytest.mark.asyncio
    async def test_delete_session(
        self,
        db_session: AsyncSession,
        chat_repo: ChatRepository,
        test_user: User,
    ):
        """Test deleting chat session"""
        # Create a session to delete
        session = await chat_repo.create_session(test_user.user_id, "To Delete")
        await db_session.commit()

        result = await chat_repo.delete_session(session.session_id)
        assert result is True

        found = await chat_repo.get_session(session.session_id)
        assert found is None

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, chat_repo: ChatRepository):
        """Test deleting non-existent session"""
        result = await chat_repo.delete_session(uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_add_message(
        self, chat_repo: ChatRepository, test_session: ChatSession
    ):
        """Test adding message to session"""
        message = await chat_repo.add_message(
            session_id=test_session.session_id,
            role="user",
            content="Hello, this is a test message",
            citations=[{"pmid": "12345", "title": "Test Paper"}],
            latency_ms=150,
        )
        assert message is not None
        assert message.content == "Hello, this is a test message"
        assert message.role == "user"

    @pytest.mark.asyncio
    async def test_add_message_auto_title(
        self,
        db_session: AsyncSession,
        chat_repo: ChatRepository,
        test_user: User,
    ):
        """Test that first user message sets session title"""
        session = await chat_repo.create_session(test_user.user_id, None)
        await db_session.commit()

        await chat_repo.add_message(
            session_id=session.session_id,
            role="user",
            content="What are the benefits of exercise?",
        )
        await db_session.commit()

        updated_session = await chat_repo.get_session(session.session_id)
        assert updated_session.title is not None
        assert "exercise" in updated_session.title.lower()

    @pytest.mark.asyncio
    async def test_get_session_messages(
        self,
        db_session: AsyncSession,
        chat_repo: ChatRepository,
        test_session: ChatSession,
    ):
        """Test getting session messages"""
        # Add messages
        await chat_repo.add_message(test_session.session_id, "user", "Question 1")
        await chat_repo.add_message(test_session.session_id, "assistant", "Answer 1")
        await db_session.commit()

        messages = await chat_repo.get_session_messages(test_session.session_id)
        assert len(messages) >= 2

    @pytest.mark.asyncio
    async def test_get_message_count(
        self,
        db_session: AsyncSession,
        chat_repo: ChatRepository,
        test_session: ChatSession,
    ):
        """Test getting message count"""
        await chat_repo.add_message(test_session.session_id, "user", "Test")
        await db_session.commit()

        count = await chat_repo.get_message_count(test_session.session_id)
        assert count >= 1


class TestUserRepository:
    """Tests for UserRepository"""

    @pytest_asyncio.fixture
    async def user_repo(self, db_session: AsyncSession):
        """Create user repository instance"""
        return UserRepository(db_session)

    @pytest.mark.asyncio
    async def test_get_by_email(
        self, user_repo: UserRepository, test_user: User
    ):
        """Test getting user by email"""
        user = await user_repo.get_by_email(test_user.email)
        assert user is not None
        assert user.email == test_user.email

    @pytest.mark.asyncio
    async def test_get_by_email_not_found(self, user_repo: UserRepository):
        """Test getting non-existent user by email"""
        user = await user_repo.get_by_email("nonexistent@example.com")
        assert user is None

    @pytest.mark.asyncio
    async def test_get_by_id(
        self, user_repo: UserRepository, test_user: User
    ):
        """Test getting user by ID"""
        user = await user_repo.get_by_id(test_user.user_id)
        assert user is not None
        assert user.email == test_user.email

    @pytest.mark.asyncio
    async def test_get_by_id_with_uuid(
        self, user_repo: UserRepository, test_user: User
    ):
        """Test getting user by UUID object"""
        from uuid import UUID
        user_uuid = UUID(test_user.user_id) if isinstance(test_user.user_id, str) else test_user.user_id
        user = await user_repo.get_by_id(user_uuid)
        assert user is not None

    @pytest.mark.asyncio
    async def test_create_user(self, user_repo: UserRepository):
        """Test creating new user"""
        user = await user_repo.create_user(
            email=f"newuser_{uuid4().hex[:8]}@example.com",
            password_hash=hash_password("password123"),
            name="New User",
            organization="Test Org",
        )
        assert user is not None
        assert user.subscription_tier == "free"

    @pytest.mark.asyncio
    async def test_create_user_with_fields(self, user_repo: UserRepository):
        """Test creating user with research fields and interests"""
        user = await user_repo.create_user(
            email=f"researcher_{uuid4().hex[:8]}@example.com",
            password_hash=hash_password("password123"),
            name="Researcher",
            organization="University",
            research_fields=["genomics", "bioinformatics"],
            interests=["cancer", "immunotherapy"],
        )
        assert user is not None
        assert user.research_fields == ["genomics", "bioinformatics"]
        assert user.interests == ["cancer", "immunotherapy"]

    @pytest.mark.asyncio
    async def test_update_last_login(
        self,
        db_session: AsyncSession,
        user_repo: UserRepository,
        test_user: User,
    ):
        """Test updating last login"""
        await user_repo.update_last_login(test_user.user_id)
        await db_session.commit()

        user = await user_repo.get_by_email(test_user.email)
        assert user.last_login is not None

    @pytest.mark.asyncio
    async def test_update_interests(
        self,
        db_session: AsyncSession,
        user_repo: UserRepository,
        test_user: User,
    ):
        """Test updating user interests"""
        interests = ["machine learning", "drug discovery", "genomics"]
        updated = await user_repo.update_interests(test_user.user_id, interests)
        await db_session.commit()

        assert updated is not None
        assert updated.interests == interests

    @pytest.mark.asyncio
    async def test_update_interests_max_limit(
        self,
        user_repo: UserRepository,
        test_user: User,
    ):
        """Test that interests are limited to 10"""
        interests = [f"interest_{i}" for i in range(15)]
        updated = await user_repo.update_interests(test_user.user_id, interests)
        assert updated is not None
        assert len(updated.interests) == 10

    @pytest.mark.asyncio
    async def test_deactivate(
        self, user_repo: UserRepository, test_user: User
    ):
        """Test deactivating user"""
        result = await user_repo.deactivate(test_user.user_id)
        assert result is True

        user = await user_repo.get_by_email(test_user.email)
        assert user.is_active is False

    @pytest.mark.asyncio
    async def test_deactivate_not_found(self, user_repo: UserRepository):
        """Test deactivating non-existent user"""
        result = await user_repo.deactivate(str(uuid4()))
        assert result is False
