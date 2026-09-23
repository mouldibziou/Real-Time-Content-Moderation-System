# Application runtime contract

The runtime provides the application foundation for the future API. It receives validated settings, builds the classification and explanation services, and coordinates access to their shared model and tokenizer. Construction is lazy: resources are loaded when `start()` is called.

## Public methods

| Purpose | Methods |
| --- | --- |
| Lifecycle | `start()`, `stop()` |
| Classification | `predict()`, `predict_batch()` |
| Explanation | `explain()`, `explain_batch()` |
| Inspection | `is_ready()`, `status()`, `model_info()` |

All nine methods are public Python methods. The future API will call `start()` and `stop()` through its application lifecycle; they are not intended to be HTTP endpoints. A public runtime method does not automatically require an endpoint.

## Lifecycle

| State | Behavior |
| --- | --- |
| `STOPPED` | Services are unavailable. `start()` begins startup; `stop()` is a no-op. |
| `STARTING` | Services are being built. Model operations and metadata access are unavailable. |
| `READY` | Model operations may be admitted, metadata is available, and shutdown may begin. Calling `start()` again is a no-op. |
| `STOPPING` | New model operations are rejected. Queued callers are woken and rejected; the active operation is allowed to finish. |

`is_ready()` returns `True` only in `READY`. Model operations with valid inputs, and `model_info()`, raise `RuntimeNotReadyError` in all other states.

Calling `start()` or `stop()` during `STARTING` or `STOPPING` raises `RuntimeError`. These calls do not wait for the lifecycle transition to complete.

Successful startup publishes the services and sets the state to `READY`. If service construction raises an exception, startup restores `STOPPED`, logs the failure, and re-raises the original exception. Startup can then be retried.

## Validation and admission

Prediction and explanation inputs are validated before checking readiness or admitting a ticket. Invalid input can therefore raise a validation error even when the runtime is not ready.

- Text must be a string, contain non-whitespace characters, and respect the character limit.
- Batches must be nonempty lists, respect the operation's batch-size limit, and contain valid texts.
- Explanation labels must be strings matching a configured label. One label applies to the entire explanation batch.

Valid inputs are preserved, including their original whitespace and batch order. Tokenization uses the configured token limit (`max_length=512`) and may truncate text that passes character validation.

There are **two queues in total**: one for classification and one for explanation. Each has its own waiting capacity. Admission rejects a caller with `RuntimeBusyError` when its queue is full. The active operation does not occupy a waiting slot. A batch occupies one ticket and one execution turn, regardless of its number of texts.

## Scheduling and synchronization

Tickets are served FIFO within each category, in enqueue order. There is no global FIFO order across both queues.

When both queues have work, classification is preferred until the consecutive-classification count reaches `classification_burst_limit`. An explanation is then selected and resets the count. If only one category has work, it can proceed without waiting for the other category. The counter resets when the runtime becomes idle with both queues empty.

Each runtime permits **one active model operation at a time** across all four prediction and explanation methods. A selected caller executes its own service call; the queues do not have dedicated worker threads.

The condition lock protects lifecycle state, queues, and model reservations. It is released during model execution. Waiting callers release the lock through `Condition.wait()` and recheck their eligibility after waking. Notifications do not guarantee which thread runs next; ticket selection enforces the scheduling policy.

Classification priority cannot interrupt an explanation already executing. Model reservations are released in `finally`, including when the service raises an exception. Service exceptions propagate unchanged.

These guarantees apply to one runtime instance in one process. Multiple application processes would have separate models, queues, and limits.

## Admission timeout

There is **one admission-wait timeout**, not an execution timeout. All four model operations use `operation_wait_timeout_seconds`.

A deadline is calculated when admission begins, after validation. Repeated wakeups do not restart it. If the caller has not claimed the model before its deadline, it raises `OperationWaitTimeoutError` and removes its queued ticket.

The timeout does not interrupt or cancel an operation that has already claimed the model. It is not a hard upper bound on total method duration: thread scheduling and lock reacquisition can also delay returning to the caller.

## Shutdown

`stop()` changes `READY` to `STOPPING` and wakes waiting callers. Those callers reject their pending operations and remove their tickets; shutdown does **not** execute all queued work.

Shutdown waits for the active operation to finish and for queued callers to clean up. It then clears the runtime's service reference, resets the scheduling counter, and enters `STOPPED`.

There is no forced cancellation or shutdown deadline. If an active operation never finishes, shutdown can wait indefinitely. Clearing the service reference does not guarantee immediate return of memory to the operating system or accelerator; other references and allocator caches may retain it.

## Status and model metadata

Inspection methods do not enter the model queues or reserve the model. They briefly acquire the condition lock and can run during prediction or explanation.

- `status()` works in every lifecycle state. It returns an immutable snapshot of the state, active-operation flag, waiting counts, and queue capacities.
- `model_info()` requires `READY`. It returns immutable metadata containing model type, resolved device, labels, thresholds, token limit, and explanation steps. It exposes neither the model object nor a filesystem path.
- `is_ready()` reports the current readiness state.

A snapshot describes the moment it was taken; it does not guarantee that a later operation will be admitted.

## Errors

| Exception | Meaning |
| --- | --- |
| `TypeError` | An input has the wrong type. |
| `ValueError` | An input has an invalid value, label, length, or batch size. |
| `RuntimeNotReadyError` | Model operations or metadata are unavailable in the current state. |
| `RuntimeBusyError` | The relevant waiting queue is full. |
| `OperationWaitTimeoutError` | The caller exhausted its wait for model access. |
| `RuntimeError` | An invalid lifecycle transition or internal ticket-selection misuse occurred. |

`RuntimeNotReadyError` and `RuntimeBusyError` inherit from `RuntimeError`. `OperationWaitTimeoutError` inherits from `TimeoutError`. Errors originating in resource loading or service execution propagate unchanged; a service's own `TimeoutError` is not automatically an admission timeout.

## Logging ownership

Current logging records startup started/completed, startup failure with its traceback, successful shutdown, and model loading started/completed with loading duration and resolved device. Lifecycle logging occurs outside the condition lock. Resource loading propagates failures to startup rather than logging a duplicate traceback.

Service logging is deferred. The future HTTP layer will own request outcomes, request IDs, total request durations, and unexpected request-failure tracebacks. Each failure should have one traceback owner. The application entry point will configure logging centrally, with module loggers identifying each layer.

Metrics are deferred until after the API layer. Queue capacities, batch limits, burst limits, and admission timeout are provisional and require workload measurements. This runtime contract does not establish a 50-requests-per-second capacity guarantee.
