import uuid
from sqlalchemy import ForeignKey, Index, Text, String, Float, Integer, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Classification(Base):
    __tablename__ = "classifications"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    model_type: Mapped[str] = mapped_column(String, nullable=False)
    device: Mapped[str] = mapped_column(String, nullable=False)
    thresholds: Mapped[dict] = mapped_column(JSONB, nullable=False)
    labels: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("text_hash", "model_version", name="uq_class_text_model"),
        Index("ix_class_created", "created_at"),
    )


class CorrectionEvent(Base):
    __tablename__ = "correction_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    classification_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("classifications.id", ondelete="CASCADE"), nullable=False)
    corrected_labels: Mapped[dict] = mapped_column(JSONB, nullable=False)
    moderator_id: Mapped[str | None] = mapped_column(String, nullable=True)
    flagged_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    created_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_corr_class", "classification_id"),
        Index("ix_corr_latest", "classification_id", "created_at"),
    )


class Explanation(Base):
    __tablename__ = "explanations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    classification_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("classifications.id", ondelete="CASCADE"), nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    spans: Mapped[list] = mapped_column(JSONB, nullable=False)
    convergence_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("classification_id", "label", "model_version", name="uq_expl_cache"),
        Index("ix_expl_class", "classification_id"),
    )


class ExplanationJob(Base):
    __tablename__ = "explanation_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    classification_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("classifications.id", ondelete="CASCADE"), nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="queued")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    lease_token: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    lease_expires_at: Mapped[object | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("classification_id", "label", "model_version", name="uq_expl_job_cache"),
        Index("ix_expl_job_status", "status"),
        Index("ix_expl_job_class", "classification_id"),
    )
