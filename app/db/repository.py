from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Classification, CorrectionEvent, Explanation

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
    await session.commit() 

async def get_classification(
        session: AsyncSession,
        *,
        classification_id: UUID
):
    """
    lookup function for the explanation endpoint.
    Returns the full row so callers get text + version + labels
    in one PK hit; handler maps None to 404.
    """
    return await session.get(Classification, classification_id)


async def save_classifications_batch(
    session: AsyncSession,
    *,
    items: list[dict],
) -> None:
    # Batch is one transaction: ids pre-generated upstream so the response
    # can return before this background commit. Single commit keeps
    # texts + results atomically together; caller chunks if huge.
    session.add_all(
        [
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
            for item in items
        ]
    )
    await session.commit()


async def find_explanation(
    session: AsyncSession,
    *,
    classification_id: UUID,
    label: str,
    model_version: str,
):
    # Cache check before GPU: hit returns instantly, miss runs the pool.
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
    await session.commit()


async def save_explanations_batch(
    session: AsyncSession,
    *,
    items: list[dict],
) -> None:
    # Multi-label future = N rows, no schema change. One commit per batch.
    session.add_all(
        [
            Explanation(
                id=item["id"],
                classification_id=item["classification_id"],
                label=item["label"],
                spans=item["spans"],
                convergence_delta=item.get("convergence_delta"),
                model_version=item["model_version"],
            )
            for item in items
        ]
    )
    await session.commit()


async def add_correction(
    session: AsyncSession,
    *,
    classification_id: UUID,
    corrected_labels: dict,
    flagged_at: datetime,
    moderator_id: str | None = None,
) -> None:
    # Append-only: history preserved, latest row per classification wins.
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
):
    # Dashboard feed: newest first, paginated. Filters join later.
    stmt = (
        select(Classification)
        .order_by(Classification.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())
