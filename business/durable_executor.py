"""
Durable outbound delivery for business writes.

The synchronous `BusinessToolExecutor` records the customer's action in
the local business database and returns immediately, which keeps the
voice turn fast. Everything downstream of that record (CRM sync,
confirmation SMS, callback dispatch, dispute ticket creation) is a side
effect that must happen exactly once even if the app restarts or a
worker crashes. Those side effects are enqueued as `business_write`
jobs in the guaranteed-delivery pipeline (see jobs/).

Sequence for a write tool:

    1. inner executor performs the local record (unchanged behaviour)
    2. a job is submitted with a deterministic idempotency key
       hash(scope, customer_id, tool, canonical arguments)
    3. the tool result carries job_id and delivery="QUEUED"
    4. the agent speaks the confirmation; the worker completes delivery

Duplicate turns (same customer, same tool, same arguments) hit the same
idempotency key and never create a second job, which complements the
session-level idempotency guard in voice/idempotency.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from jobs import JobStore

from .schemas import ToolExecutionResult

logger = logging.getLogger(__name__)

WRITE_TOOLS = {
    "record_promise_to_pay",
    "schedule_callback",
    "record_payment_reported",
    "open_dispute",
    "record_financial_hardship",
    "request_human_escalation",
}


def idempotency_key_for(
    scope: str, customer_id: str, tool: str, arguments: dict[str, Any]
) -> str:
    """
    scope        deployment/app instance (keeps keys unique across tenants)
    customer_id  the account the write belongs to
    tool + canonical arguments
    """
    canonical = json.dumps(arguments or {}, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(
        f"{scope}|{customer_id}|{tool}|{canonical}".encode()
    ).hexdigest()[:24]
    return f"bw-{tool}-{digest}"


class DurableWriteExecutor:
    """Decorator around any executor exposing execute(tool, arguments, user_text, customer_id)."""

    def __init__(self, inner: Any, store: JobStore, *, scope: str = "app") -> None:
        self.inner = inner
        self.store = store
        self.scope = scope

    def __getattr__(self, name: str) -> Any:  # delegate everything else
        return getattr(self.inner, name)

    def execute(
        self,
        tool: str,
        arguments: dict[str, Any] | None,
        user_text: str,
        customer_id: str,
    ) -> ToolExecutionResult:
        result = self.inner.execute(tool, arguments, user_text, customer_id)

        if tool not in WRITE_TOOLS or not result.success:
            return result

        key = idempotency_key_for(self.scope, customer_id, tool, arguments or {})
        job, created = self.store.submit(
            idempotency_key=key,
            type="business_write",
            payload={
                "tool": tool,
                "arguments": arguments or {},
                "customer_id": customer_id,
                "scope": self.scope,
                "local_record": result.data,
            },
        )

        logger.info(
            "DURABLE WRITE | tool=%s | job_id=%s | status=%s | created=%s",
            tool, job.job_id, job.status.value, created,
        )

        data = dict(result.data)
        data["delivery"] = {
            "job_id": job.job_id,
            "status": job.status.value,
            "idempotency_key": key,
            "deduplicated": not created,
        }
        return result.model_copy(update={"data": data})
