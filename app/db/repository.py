from datetime import datetime
from logging import getLogger
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Classification, CorrectionEvent, Explanation, ExplanationJob

logger = getLogger(__name__)

async def save_classification(
        session: AsyncSession,
        *,
        id: UUID,
        text: str,
        text_hash: str,
        model_version: str,
        model_type: str,
        device: str,
        thresholds: dict,
        labels: dict,
):
    """Persist one classify result after the response returned.

    Use case: POST /v1/classify return-first path. Handler generates id
    upfront, returns {id, text, labels}, then background task calls this.
    Duplicates on (text_hash, model_version) are benign repeats: roll back
    and keep the earliest row instead of raising.
    """
    # Keyword construction only; server owns created_at via now().
    new_classification = Classification(
        id=id,
        text=text,
        text_hash=text_hash,
        model_version=model_version,
        model_type=model_type,
        device=device,
        thresholds=thresholds,
        labels=labels,
    )
    session.add(new_classification)
    # without await we return a coroutine and nothing persists.
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        logger.info(
            "duplicate classification ignored for hash %s model %s",
            text_hash,
            model_version,
        ) 

async def get_classification(
        session: AsyncSession,
        *,
        classification_id: UUID
):
    """Fetch one classification by PK for strict explain and detail views.

    Use case: POST /v1/explain takes classification_id, loads the exact
    stored text to run inference on; GET detail reuses the same row.
    Returns full row or None (route maps None to 404).
    """
    return await session.get(Classification, classification_id)


async def save_classifications_batch(
    session: AsyncSession,
    *,
    items: list[dict],
) -> None:
    """Persist one classify_batch result set in a single transaction.

    Use case: POST /v1/classify_batch return-first path. Ids pre-generated
    so the response returns before this background commit. Duplicate
    (hash, version) rows are skipped per item so one repeat never fails
    the whole batch.
    """
    for item in items:
        session.add(
            Classification(
                id=item["id"],
                text=item["text"],
                text_hash=item["text_hash"],
                model_version=item["model_version"],
                model_type=item["model_type"],
                device=item["device"],
                thresholds=item["thresholds"],
                labels=item["labels"],
            )
        )
        try:
            await session.flush()
        except IntegrityError:
            await session.rollback()
            logger.info(
                "duplicate classification ignored for hash %s",
                item["text_hash"],
            )
            continue
    await session.commit()


async def find_explanation(
    session: AsyncSession,
    *,
    classification_id: UUID,
    label: str,
    model_version: str,
):
    """Cache check before GPU work for strict explain.

    Use case: POST /v1/explain hits this first; hit returns spans
    instantly, miss runs the pool then save_explanation.
    """
    stmt = (
        select(Explanation)
        .where(Explanation.classification_id == classification_id)
        .where(Explanation.label == label)
        .where(Explanation.model_version == model_version)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def save_explanation(
    session: AsyncSession,
    *,
    id: UUID,
    classification_id: UUID,
    label: str,
    spans: list,
    convergence_delta: float | None,
    model_version: str,
) -> None:
    """Persist one explain miss after the response returned.

    Use case: POST /v1/explain cache-miss path, background write so
    repeat requests become cache hits. Concurrent misses for the same
    triple are benign: keep the first winner, ignore the loser.
    """
    session.add(
        Explanation(
            id=id,
            classification_id=classification_id,
            label=label,
            spans=spans,
            convergence_delta=convergence_delta,
            model_version=model_version,
        )
    )
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        logger.info(
            "duplicate explanation ignored for %s %s %s",
            classification_id,
            label,
            model_version,
        )


async def save_explanations_batch(
    session: AsyncSession,
    *,
    items: list[dict],
) -> None:
    """Persist one explain_batch miss set in a single transaction.

    Use case: POST /v1/explain_batch return-first path. Future multi-label
    is N rows here with no schema change. Per-triple duplicates are
    skipped so one race never fails the batch.
    """
    for item in items:
        session.add(
            Explanation(
                id=item["id"],
                classification_id=item["classification_id"],
                label=item["label"],
                spans=item["spans"],
                convergence_delta=item.get("convergence_delta"),
                model_version=item["model_version"],
            )
        )
        try:
            await session.flush()
        except IntegrityError:
            await session.rollback()
            logger.info(
                "duplicate explanation ignored for %s %s",
                item["classification_id"],
                item["label"],
            )
            continue
    await session.commit()


async def add_correction(
    session: AsyncSession,
    *,
    classification_id: UUID,
    corrected_labels: dict,
    flagged_at: datetime,
    moderator_id: str | None = None,
) -> None:
    """Append one moderator correction event, never update in place.

    Use case: future POST /v1/classifications/{id}/corrections.
    Latest row per classification wins for display; history preserved.
    """
    session.add(
        CorrectionEvent(
            classification_id=classification_id,
            corrected_labels=corrected_labels,
            flagged_at=flagged_at,
            moderator_id=moderator_id,
        )
    )
    await session.commit()


async def list_classifications(
    session: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    corrected: bool | None = None,
):
    """Paginated newest-first feed for the moderator dashboard.

    Use case: GET /v1/classifications list. corrected=None returns all,
    True only corrected, False only uncorrected (queue view). Filter
    pushed to SQL via EXISTS so pages stay exact.
    """
    stmt = select(Classification)
    if corrected is True:
        stmt = stmt.where(
            select(CorrectionEvent.id)
            .where(CorrectionEvent.classification_id == Classification.id)
            .exists()
        )
    elif corrected is False:
        stmt = stmt.where(
            ~select(CorrectionEvent.id)
            .where(CorrectionEvent.classification_id == Classification.id)
            .exists()
        )
    stmt = stmt.order_by(Classification.created_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_latest_correction(
    session: AsyncSession,
    *,
    classification_id: UUID,
):
    """Latest correction wins for display; None means uncorrected.

    Use case: detail and list overlay without mutating history.
    """
    stmt = (
        select(CorrectionEvent)
        .where(CorrectionEvent.classification_id == classification_id)
        .order_by(CorrectionEvent.created_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def list_latest_corrections(
    session: AsyncSession,
    *,
    classification_ids: list[UUID],
):
    """Batch latest-correction lookup to avoid N+1 on list pages.

    Use case: GET list overlays corrected flags in two queries total.
    Returns map of classification_id string to row.
    """
    if not classification_ids:
        return {}
    stmt = (
        select(CorrectionEvent)
        .where(CorrectionEvent.classification_id.in_(classification_ids))
        .order_by(CorrectionEvent.classification_id, CorrectionEvent.created_at.desc())
    )
    result = await session.execute(stmt)
    rows = list(result.scalars().all())
    latest: dict[str, object] = {}
    for row in rows:
        key = str(row.classification_id)
        if key not in latest:
            latest[key] = row
    return latest


async def list_explanations(
    session: AsyncSession,
    *,
    classification_id: UUID,
):
    """All cached explanations for one classification across labels/versions.

    Use case: detail view shows which labels are already explained.
    """
    stmt = (
        select(Explanation)
        .where(Explanation.classification_id == classification_id)
        .order_by(Explanation.created_at.desc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_job(
    session: AsyncSession,
    *,
    job_id: UUID,
):
    """Fetch one job by PK for status polling.

    Use case: GET /v1/jobs/{id}. Returns row or None (route 404s).
    """
    return await session.get(ExplanationJob, job_id)


async def find_job(
    session: AsyncSession,
    *,
    classification_id: UUID,
    label: str,
    model_version: str,
):
    """Idempotent lookup for repeats: same triple returns existing job.

    Use case: POST /explain creates-or-returns instead of doubling GPU.
    """
    stmt = (
        select(ExplanationJob)
        .where(ExplanationJob.classification_id == classification_id)
        .where(ExplanationJob.label == label)
        .where(ExplanationJob.model_version == model_version)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def create_job(
    session: AsyncSession,
    *,
    classification_id: UUID,
    label: str,
    model_version: str,
):
    """Insert a queued job; on unique race return the existing winner.

    Use case: POST /explain miss path. Returns (row, created_bool).
    """
    row = ExplanationJob(
        classification_id=classification_id,
        label=label,
        model_version=model_version,
        status="queued",
    )
    session.add(row)
    try:
        await session.commit()
        await session.refresh(row)
        return row, True
    except IntegrityError:
        await session.rollback()
        existing = await find_job(
            session,
            classification_id=classification_id,
            label=label,
            model_version=model_version,
        )
        assert existing is not None
        return existing, False


async def mark_job(
    session: AsyncSession,
    *,
    job_id: UUID,
    status: str,
    error: str | None = None,
):
    """Transition running|done|failed with updated_at bump.

    Use case: worker owns transitions; API never calls this.
    """
    row = await session.get(ExplanationJob, job_id)
    if row is None:
        return None
    row.status = status
    row.error = error
    from datetime import timezone

    row.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(row)
    return row
