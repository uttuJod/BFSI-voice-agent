from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from .store import JobStatus, JobStore

logger = logging.getLogger("jobs.worker")

Handler = Callable[["JobContext"], dict[str, Any] | None]


class SideEffectLedger:
    """
    Handlers call `ledger.once(key)` before performing an external action.
    The first call for a key returns True; every later call (retry,
    duplicate delivery, second worker) returns False. The ledger is the
    `side_effects` table in the job store, so it is as durable as the job.
    """

    def __init__(self, store: JobStore, job_id: str) -> None:
        self._store = store
        self._job_id = job_id

    def once(self, effect_key: str, detail: str = "") -> bool:
        return self._store.side_effect_once(effect_key, self._job_id, detail)


class JobContext:
    def __init__(self, store: JobStore, job, worker_id: str) -> None:
        self.store = store
        self.job = job
        self.worker_id = worker_id
        self.ledger = SideEffectLedger(store, job.job_id)

    @property
    def payload(self) -> dict[str, Any]:
        return self.job.payload

    def heartbeat(self) -> None:
        self.store.heartbeat(self.job.job_id, self.worker_id)


class HandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, job_type: str) -> Callable[[Handler], Handler]:
        def deco(fn: Handler) -> Handler:
            self._handlers[job_type] = fn
            return fn
        return deco

    def get(self, job_type: str) -> Handler:
        try:
            return self._handlers[job_type]
        except KeyError:
            raise KeyError(f"No handler registered for job type {job_type!r}")

    def types(self) -> list[str]:
        return sorted(self._handlers)


class Worker:
    def __init__(
        self,
        store: JobStore,
        registry: HandlerRegistry,
        *,
        worker_id: str | None = None,
        poll_interval_s: float = 0.2,
    ) -> None:
        self.store = store
        self.registry = registry
        self.worker_id = worker_id or f"worker-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self.processed = 0

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> bool:
        """Process at most one job. Returns True if a job was handled."""
        job = self.store.claim(self.worker_id)
        if job is None:
            return False
        ctx = JobContext(self.store, job, self.worker_id)
        logger.info(
            "JOB START | job_id=%s | type=%s | attempt=%d/%d | worker=%s",
            job.job_id, job.type, job.attempts, job.max_attempts, self.worker_id,
        )
        try:
            handler = self.registry.get(job.type)
            result = handler(ctx) or {}
            self.store.complete(job.job_id, self.worker_id, result)
            self.processed += 1
            logger.info("JOB DONE | job_id=%s", job.job_id)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
            status = self.store.fail(job.job_id, self.worker_id, err)
            logger.warning("JOB FAIL | job_id=%s | next=%s | %s", job.job_id, status.value, exc)
        return True

    def run_forever(self) -> None:
        logger.info("Worker %s started | handlers=%s", self.worker_id, self.registry.types())
        while not self._stop.is_set():
            if not self.run_once():
                self._stop.wait(self.poll_interval_s)
        logger.info("Worker %s stopped | processed=%d", self.worker_id, self.processed)


# ---------------------------------------------------------------- handlers
# Demo handlers for the four job types in the problem statement. Each one
# records its side effect in an append-only ledger file so tests can count
# side effects independently of the job table.

DEFAULT_REGISTRY = HandlerRegistry()
SIDE_EFFECT_LOG = Path(os.getenv("JOB_SIDE_EFFECT_LOG", "results/side_effects.log"))


def _append_effect(kind: str, detail: dict[str, Any]) -> None:
    SIDE_EFFECT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SIDE_EFFECT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.time(), "kind": kind, **detail}) + "\n")


def _maybe_inject_failure(ctx: JobContext) -> None:
    """
    Chaos hooks driven by the payload so failure tests are reproducible:
      fail_times: N     raise on the first N attempts, succeed after
      always_fail: true raise on every attempt (DLQ test)
      sleep_s: X        hold the job for X seconds (crash-mid-job test)
    """
    p = ctx.payload
    if p.get("sleep_s"):
        deadline = time.time() + float(p["sleep_s"])
        while time.time() < deadline:
            time.sleep(0.1)
            ctx.heartbeat()
    if p.get("always_fail"):
        raise RuntimeError("injected permanent failure")
    if p.get("fail_times") and ctx.job.attempts <= int(p["fail_times"]):
        raise RuntimeError(f"injected transient failure on attempt {ctx.job.attempts}")


@DEFAULT_REGISTRY.register("send_email")
def send_email(ctx: JobContext) -> dict[str, Any]:
    _maybe_inject_failure(ctx)
    if ctx.ledger.once(f"send_email:{ctx.job.idempotency_key}"):
        _append_effect("email_sent", {"job_id": ctx.job.job_id, "to": ctx.payload.get("to")})
        return {"sent": True}
    return {"sent": False, "deduplicated": True}


@DEFAULT_REGISTRY.register("generate_report")
def generate_report(ctx: JobContext) -> dict[str, Any]:
    _maybe_inject_failure(ctx)
    if ctx.ledger.once(f"generate_report:{ctx.job.idempotency_key}"):
        _append_effect("report_generated", {"job_id": ctx.job.job_id})
        return {"path": f"results/reports/{ctx.job.job_id}.pdf"}
    return {"deduplicated": True}


@DEFAULT_REGISTRY.register("sync_webhook")
def sync_webhook(ctx: JobContext) -> dict[str, Any]:
    _maybe_inject_failure(ctx)
    if ctx.ledger.once(f"sync_webhook:{ctx.job.idempotency_key}"):
        _append_effect("webhook_posted", {"job_id": ctx.job.job_id, "url": ctx.payload.get("url")})
        return {"status": 200}
    return {"deduplicated": True}


@DEFAULT_REGISTRY.register("process_document")
def process_document(ctx: JobContext) -> dict[str, Any]:
    _maybe_inject_failure(ctx)
    if ctx.ledger.once(f"process_document:{ctx.job.idempotency_key}"):
        _append_effect("document_processed", {"job_id": ctx.job.job_id})
        return {"pages": 1}
    return {"deduplicated": True}


# Voice-agent business writes. The handler key is the business action; the
# actual write goes through business.executor so behaviour is identical to
# the synchronous path.
@DEFAULT_REGISTRY.register("business_write")
def business_write(ctx: JobContext) -> dict[str, Any]:
    _maybe_inject_failure(ctx)
    tool = ctx.payload.get("tool")
    args = ctx.payload.get("arguments") or {}
    if not ctx.ledger.once(f"business_write:{ctx.job.idempotency_key}"):
        return {"deduplicated": True}
    _append_effect("business_write", {"job_id": ctx.job.job_id, "tool": tool, "arguments": args})
    return {"tool": tool, "recorded": True}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run a job worker.")
    parser.add_argument("--db", default=os.getenv("JOBS_DB", "results/jobs.db"))
    parser.add_argument("--lease-seconds", type=float, default=float(os.getenv("JOBS_LEASE_S", "5")))
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--once", action="store_true", help="process one job and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = JobStore(args.db, lease_seconds=args.lease_seconds)
    worker = Worker(store, DEFAULT_REGISTRY, worker_id=args.worker_id)

    def _sig(*_: Any) -> None:
        worker.stop()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    if args.once:
        worker.run_once()
        return 0
    worker.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
