"""relabel yahoo splits as split_or_bonus

Revision ID: 2df771dcbba7
Revises: 409e4e5eede2
Create Date: 2026-11-25 04:28:27.687275
"""
from collections.abc import Sequence

from alembic import op


revision: str = '2df771dcbba7'
down_revision: str | None = '409e4e5eede2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Yahoo can't tell splits from bonuses (tax treatment differs), so mark them ambiguous.
    op.execute(
        "UPDATE corporate_actions SET kind = 'SPLIT_OR_BONUS' "
        "WHERE source = 'yahoo' AND kind = 'SPLIT'"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE corporate_actions SET kind = 'SPLIT' "
        "WHERE source = 'yahoo' AND kind = 'SPLIT_OR_BONUS'"
    )
