"""
Guaranteed-delivery job pipeline for side-effecting business actions.

Why it exists
-------------
The voice agent records promises to pay, schedules callbacks, opens
disputes and requests escalations. Those are side effects that must
happen exactly once even when a worker crashes mid-job, the network
retries a submission, or a message is delivered twice. The voice path
must also stay fast: the agent confirms as soon as the job is durably
QUEUED, not when it completes.

Design
------
* Single durable store (SQLite in WAL mode) acting as both job table and
  queue. A job row is the unit of delivery; there is no separate broker
  to fall out of sync with. "Broker restart" therefore means restarting
  the process that owns the database file, and the queue survives
  because it is the database.
* Idempotency: UNIQUE(idempotency_key). A duplicate POST returns the
  original job. A worker claims a job with a lease (visibility timeout);
  if the worker dies the lease expires and another worker reclaims the
  job. Handlers are given the job_id so the side effect itself can be
  deduplicated in the side-effect store (see `SideEffectLedger`).
* Retries: exponential backoff with jitter, configurable max attempts.
* DLQ: after max attempts the job is marked DLQ with the last error and
  stack excerpt, and listed by GET /v1/jobs/dlq.
* Observability: every transition is appended to job_events with a
  correlation id; GET /v1/jobs/{id} returns status, attempts, timings
  and the error; GET /v1/jobs/stats returns counts by state.
"""

from .store import JobStore, JobStatus, RetryPolicy
from .worker import Worker, SideEffectLedger, HandlerRegistry

__all__ = [
    "JobStore",
    "JobStatus",
    "RetryPolicy",
    "Worker",
    "SideEffectLedger",
    "HandlerRegistry",
]
