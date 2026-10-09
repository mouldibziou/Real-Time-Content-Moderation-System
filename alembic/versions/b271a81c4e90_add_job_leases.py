"""Add recoverable explanation job ownership."""
from alembic import op
import sqlalchemy as sa

revision = "b271a81c4e90"
down_revision = "53223f4cc16d"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("explanation_jobs", sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("explanation_jobs", sa.Column("lease_token", sa.UUID(), nullable=True))
    op.add_column("explanation_jobs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))


def downgrade():
    op.drop_column("explanation_jobs", "lease_expires_at")
    op.drop_column("explanation_jobs", "lease_token")
    op.drop_column("explanation_jobs", "attempts")
