# Explanation worker Redis connections

The explanation worker uses its own Redis client. The API's enqueue client
and the job storage contract are unchanged.

The worker uses a five-second blocking pop, a 15-second socket read timeout,
and a three-second connection timeout. An empty queue therefore returns
normally rather than racing the read timeout. `REDIS_URL` query parameters
can override the socket timeouts; the worker rejects non-finite or disabled
timeouts, read timeouts of five seconds or less, and non-positive connection
timeouts before loading the model.

Connection failures and read timeouts are retried five times after the initial
attempt. Delays are 0.5, 1, 2, 4, and 8 seconds. Redis's implicit retries are
disabled for this client so the two policies cannot multiply. Each successful
command resets the retry budget. Startup requeue pushes, blocking pops, and
batch-drain pops all use this policy.

The `expl_worker` logger reports connection health transitions:

- `worker_health=ready`: a Redis command succeeded, including an empty pop.
- `worker_health=degraded`: a connection failure is being retried.
- `worker_health=failed`: the retry budget was exhausted.
- `worker_health=stopped`: worker cleanup completed.

These are connection health logs, not an HTTP readiness probe or a periodic
worker heartbeat. The worker exits with an error after retry exhaustion; run
it under a supervisor configured to restart failed processes. SIGTERM wakes
retry backoff immediately. A pending Redis command remains bounded by its
configured timeout. The worker closes its Redis client during shutdown.

This change does not make destructive Redis pops reliable. A connection can
fail after Redis removes a job but before the worker receives the reply.
Job leases, abandoned-job recovery, durable enqueue delivery, and retrying
failed explanation computation remain separate work.

Run the targeted checks with:

```sh
venv/bin/python -m unittest tests.test_worker_redis -v
```
