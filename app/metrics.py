"""Bounded, process-local Prometheus metrics for the API and worker."""
import asyncio
from contextvars import ContextVar
import logging
from threading import Lock

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily

from app.observability import log_event

LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
HTTP_BUCKETS = (.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30)
JOB_BUCKETS = (.1, .5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800)
_api_metrics = ContextVar("api_metrics", default=None)
logger = logging.getLogger(__name__)


class ApiMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.classification_requests = Gauge(
            "classification_requests_current", "Accepted classification calls waiting for or executing model inference; batches count as one call.",
            ("state",), registry=self.registry)
        for state in ("queued", "running"):
            self.classification_requests.labels(state)
        self._queue_lock = Lock()
        self._queue_counts = None
        self.requests = Counter("http_requests_total", "Completed HTTP requests, including error responses.",
                                ("method", "route", "status_code"), registry=self.registry)
        self.duration = Histogram("http_request_duration_seconds", "Application request handling duration in seconds.",
                                  ("method", "route"), buckets=HTTP_BUCKETS, registry=self.registry)
        self.items = Counter("classification_items_total", "Successfully classified and persisted input texts.",
                             registry=self.registry)
        self.predictions = Counter("classification_predictions_total", "Persisted classification decisions per toxicity label.",
                                   ("label", "decision"), registry=self.registry)
        for label in LABELS:
            for decision in ("positive", "negative"):
                self.predictions.labels(label, decision)
        self.jobs_created = Counter("explanation_jobs_created_total", "New explanation jobs committed by this API process.",
                                    registry=self.registry)
        self.cache_hits = Counter("explanation_cache_hits_total", "Cached explanation results served by this API process.",
                                  registry=self.registry)

    def set_queue_counts(self, counts):
        with self._queue_lock:
            self._queue_counts = counts

    def queue_counts(self):
        with self._queue_lock:
            return self._queue_counts

    def classification_ticket(self):
        return ClassificationTicket(self)

    def record_predictions(self, results):
        self.items.inc(len(results))
        for result in results:
            for label, prediction in result.labels.items():
                if label in LABELS:
                    self.predictions.labels(label, "positive" if prediction.predicted else "negative").inc()


class ClassificationTicket:
    """Track the actual executor queue, including cancellation and submission failure."""
    def __init__(self, metrics):
        self.gauge = metrics.classification_requests
        self.lock = Lock()
        self.state = "queued"
        self.gauge.labels("queued").inc()

    def start(self):
        with self.lock:
            self.gauge.labels("queued").dec()
            self.gauge.labels("running").inc()
            self.state = "running"

    def finish(self):
        with self.lock:
            if self.state != "finished":
                self.gauge.labels(self.state).dec()
                self.state = "finished"

    def cancelled(self):
        # Cancelling the waiter does not stop an inference already executing.
        with self.lock:
            if self.state == "queued":
                self.gauge.labels("queued").dec()
                self.state = "finished"


async def refresh_queue_metrics(metrics, interval=5):
    """Refresh database-backed gauges outside the dedicated scrape thread."""
    while True:
        metrics.set_queue_counts(await read_queue_counts())
        await asyncio.sleep(interval)


def current_api_metrics():
    return _api_metrics.get()


class WorkerMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.attempts = Counter("explanation_job_attempts_total", "Successfully committed worker job claims.", registry=self.registry)
        self.attempt_duration = Histogram("explanation_job_attempt_duration_seconds", "Claimed attempt handling duration, including persistence.",
                                          buckets=JOB_BUCKETS, registry=self.registry)
        self.queue_wait = Histogram("explanation_job_queue_wait_seconds", "Job creation to first claim, using database timestamps.",
                                    buckets=JOB_BUCKETS, registry=self.registry)
        self.end_to_end = Histogram("explanation_job_end_to_end_duration_seconds", "Job creation to successful result commit, including retries.",
                                    buckets=JOB_BUCKETS, registry=self.registry)
        self.outcomes = Counter("explanation_job_outcomes_total", "Committed worker job transitions, including lease recovery.",
                                ("outcome",), registry=self.registry)
        for outcome in ("completed", "retry", "failed"):
            self.outcomes.labels(outcome)


WORKER_METRICS = WorkerMetrics()


async def read_queue_counts():
    """Read shared job state once per scrape; never infer it from Redis length."""
    from sqlalchemy import func, select
    from app.db.engine import async_session
    from app.db.models import ExplanationJob
    try:
        async with asyncio.timeout(2):
            async with async_session() as session:
                result = await session.execute(
                    select(ExplanationJob.status, func.count()).where(
                        ExplanationJob.status.in_(("queued", "running"))
                    ).group_by(ExplanationJob.status)
                )
                return {"queued": 0, "running": 0, **dict(result.all())}
    except Exception:
        log_event(logger, "metrics.queue_collection_failed", level=logging.WARNING, exc_info=True)
        return None


def api_export_registry(metrics, get_queue_counts):
    class Snapshot:
        def collect(self):
            yield from metrics.registry.collect()
            queue_counts = get_queue_counts()
            success = GaugeMetricFamily("explanation_queue_collection_success", "Whether this scrape could read shared job state.")
            success.add_metric([], int(queue_counts is not None))
            yield success
            if queue_counts is not None:
                queue = GaugeMetricFamily("explanation_jobs_current", "Current shared jobs from PostgreSQL; not per replica.", labels=["status"])
                for status in ("queued", "running"):
                    queue.add_metric([status], queue_counts[status])
                yield queue
    registry = CollectorRegistry()
    registry.register(Snapshot())
    return registry


def render_api_metrics(metrics, queue_counts):
    return generate_latest(api_export_registry(metrics, lambda: queue_counts))


class MetricsMiddleware:
    def __init__(self, app, metrics):
        self.app, self.metrics = app, metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        from time import monotonic
        started = monotonic()
        original_path = scope["path"]
        status = None
        completed = False
        failed = False
        token = _api_metrics.set(self.metrics)

        async def observed_send(message):
            nonlocal status, completed
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                completed = True

        try:
            await self.app(scope, receive, observed_send)
        except Exception:
            failed = True
            raise
        finally:
            _api_metrics.reset(token)
            excluded = original_path in ("/metrics", "/metrics/", "/v1/health/live", "/v1/health/ready")
            if not excluded and (completed or failed):
                route = getattr(scope.get("route"), "path", "unmatched")
                method = scope["method"] if scope["method"] in {
                    "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"
                } else "OTHER"
                self.metrics.requests.labels(method, route, str(status if status is not None else 500)).inc()
                self.metrics.duration.labels(method, route).observe(monotonic() - started)
