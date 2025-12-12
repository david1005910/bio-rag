"""Initial migration

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users table
    op.create_table(
        "users",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("organization", sa.String(255), nullable=True),
        sa.Column("research_fields", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("interests", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("subscription_tier", sa.String(20), nullable=False, server_default="free"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_login", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.PrimaryKeyConstraint("user_id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_email", "users", ["email"])

    # Papers table
    op.create_table(
        "papers",
        sa.Column("pmid", sa.String(20), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("authors", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("journal", sa.String(255), nullable=True),
        sa.Column("publication_date", sa.DateTime(), nullable=True),
        sa.Column("doi", sa.String(100), nullable=True),
        sa.Column("keywords", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("mesh_terms", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("citation_count", sa.Integer(), nullable=True),
        sa.Column("pdf_url", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("pmid"),
    )
    op.create_index("ix_papers_publication_date", "papers", ["publication_date"])
    op.create_index("ix_papers_journal", "papers", ["journal"])

    # Saved papers table
    op.create_table(
        "saved_papers",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("pmid", sa.String(20), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("saved_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["pmid"], ["papers.pmid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id", "pmid"),
    )

    # Chat sessions table
    op.create_table(
        "chat_sessions",
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])

    # Chat messages table
    op.create_table(
        "chat_messages",
        sa.Column("message_id", sa.UUID(), nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", postgresql.JSON(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.session_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("message_id"),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])

    # Search logs table
    op.create_table(
        "search_logs",
        sa.Column("log_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("filters", postgresql.JSON(), nullable=True),
        sa.Column("result_count", sa.Integer(), nullable=True),
        sa.Column("query_time_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("log_id"),
    )
    op.create_index("ix_search_logs_user_id", "search_logs", ["user_id"])


def downgrade() -> None:
    op.drop_table("search_logs")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("saved_papers")
    op.drop_table("papers")
    op.drop_table("users")
