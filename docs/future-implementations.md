# Future implementations

## Mixed-label batched explanations

Add an explicit production API for explaining a batch in which each text may target a different label.

The API should accept paired inputs, for example `[(text, label), ...]`, validate every label, and return results in input order. Internally, group requests by label and execute one batched Integrated Gradients call per label group. This preserves the current attribution semantics: each IG calculation targets exactly one output logit.

Before implementation, decide request limits, batching and memory policy, error behavior for invalid labels, and whether the runtime exposes the capability only for flagged labels or also for arbitrary labels.

## Durable return-first persistence

Current v1 persists via FastAPI BackgroundTasks after the response returns.
On DB failure it logs and drops the write (accepted loss window for p95).

Before prod ship, choose one durability contract and implement it:

1. Degraded-serve with retry (recommended for minimum latency): keep
   serving 200s while Postgres is down. Background tasks retry with
   backoff, then park failed payloads in a durable outbox or dead-letter
   table plus a metric/alert. Lifespan DB check stays soft (log-only).
   Readiness gates on model only, DB state surfaces in /status and logs.

2. Strict durability: fail boot and pull ready when persistence is
   unhealthy. No loss, but DB blips take down classification. Lifespan
   DB check becomes hard (raise), readiness gates on DB + model.

Do not ship log-and-drop. Current logging is a placeholder proving
latency separation, not a durability guarantee.
