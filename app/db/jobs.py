from datetime import timedelta
from uuid import uuid4

from sqlalchemy import case, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert

from app.db.models import Explanation, ExplanationJob

MAX_ATTEMPTS = 3


async def recover_jobs(session):
    expired = or_(ExplanationJob.lease_expires_at.is_(None), ExplanationJob.lease_expires_at <= func.now())
    await session.execute(
        update(ExplanationJob).where(ExplanationJob.status == "running", expired).values(
            status=case((ExplanationJob.attempts >= MAX_ATTEMPTS, "failed"), else_="queued"),
            lease_token=None, lease_expires_at=None, error="Worker lease expired",
            updated_at=func.now(),
        )
    )
    await session.commit()


async def claim_job(session, job_id, *, lease_seconds=60):
    token = uuid4()
    result = await session.execute(
        update(ExplanationJob).where(
            ExplanationJob.id == job_id,
            ExplanationJob.status == "queued",
            ExplanationJob.attempts < MAX_ATTEMPTS,
        ).values(
            status="running", attempts=ExplanationJob.attempts + 1,
            lease_token=token, lease_expires_at=func.now() + timedelta(seconds=lease_seconds),
            error=None, updated_at=func.now(),
        ).returning(ExplanationJob)
    )
    job = result.scalar_one_or_none()
    await session.commit()
    return job


async def renew_job(session, job_id, token, *, lease_seconds=60):
    result = await session.execute(
        update(ExplanationJob).where(
            ExplanationJob.id == job_id, ExplanationJob.status == "running",
            ExplanationJob.lease_token == token,
        ).values(lease_expires_at=func.now() + timedelta(seconds=lease_seconds), updated_at=func.now())
    )
    await session.commit()
    return result.rowcount == 1


async def finish_job(session, job_id, token, result):
    job = (await session.execute(
        select(ExplanationJob).where(
            ExplanationJob.id == job_id, ExplanationJob.status == "running",
            ExplanationJob.lease_token == token,
        ).with_for_update()
    )).scalar_one_or_none()
    if job is None:
        return False
    spans = [dict(text=s.text, start=s.start, end=s.end, score=s.score,
                  normalized_importance=s.normalized_importance) for s in result.spans]
    await session.execute(insert(Explanation).values(
        classification_id=job.classification_id, label=job.label, model_version=job.model_version,
        spans=spans, convergence_delta=result.convergence_delta,
    ).on_conflict_do_nothing(constraint="uq_expl_cache"))
    job.status, job.error = "done", None
    job.lease_token, job.lease_expires_at = None, None
    job.updated_at = func.now()
    await session.commit()
    return True


async def fail_job(session, job_id, token, error, *, permanent=False):
    await session.execute(
        update(ExplanationJob).where(
            ExplanationJob.id == job_id, ExplanationJob.status == "running",
            ExplanationJob.lease_token == token,
        ).values(
            status="failed" if permanent else case(
                (ExplanationJob.attempts >= MAX_ATTEMPTS, "failed"), else_="queued"),
            error=error[:500], lease_token=None, lease_expires_at=None, updated_at=func.now(),
        )
    )
    await session.commit()
