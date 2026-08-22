from __future__ import annotations

from contextlib import contextmanager
import json
import os
import random
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class JobStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    DLQ = "DLQ"


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.2

    def delay_for(self, attempt: int) -> float:
        raw = min(self.max_delay_s, self.base_delay_s * (self.multiplier ** max(attempt - 1, 0)))
        return raw * (1.0 + random.uniform(-self.jitter, self.jitter))


@dataclass
class Job:
    job_id: str
    idempotency_key: str
    type: str
    payload: dict[str, Any]
    status: JobStatus
    attempts: int
    max_attempts: int
    created_at: float
    updated_at: float
    available_at: float
    lease_expires_at: float | None
    worker_id: str | None
    completed_at: float | None
    error: str | None
    result: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "idempotency_key": self.idempotency_key,
            "type": self.type,
            "payload": self.payload,
            "status": self.status.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "available_at": self.available_at,
            "lease_expires_at": self.lease_expires_at,
            "worker_id": self.worker_id,
            "completed_at": self.completed_at,
            "error": self.error,
            "result": self.result,
        }


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    available_at REAL NOT NULL,
    lease_expires_at REAL,
    worker_id TEXT,
    completed_at REAL,
    error TEXT,
    result TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_available ON jobs(status, available_at);
CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    ts REAL NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    worker_id TEXT,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id);
CREATE TABLE IF NOT EXISTS side_effects (
    effect_key TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    ts REAL NOT NULL,
    detail TEXT
);
"""


class JobStore:
    """
    Durable job table + queue on SQLite (WAL). Safe for multiple worker
    processes on one host; claims are atomic UPDATE ... WHERE status=QUEUED.
    """

    def __init__(
        self,
        path: str | os.PathLike = "results/jobs.db",
        *,
        retry_policy: RetryPolicy | None = None,
        lease_seconds: float = 30.0,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retry_policy = retry_policy or RetryPolicy()
        self.lease_seconds = lease_seconds
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ conn
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self.path, timeout=30, isolation_level=None, check_same_thread=False
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def _event(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        from_status: str | None,
        to_status: str,
        worker_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        conn.execute(
            "INSERT INTO job_events(job_id, ts, from_status, to_status, worker_id, detail) "
            "VALUES (?,?,?,?,?,?)",
            (job_id, time.time(), from_status, to_status, worker_id, detail),
        )

    # ---------------------------------------------------------------- submit
    def submit(
        self,
        *,
        idempotency_key: str,
        type: str,
        payload: dict[str, Any],
        max_attempts: int | None = None,
    ) -> tuple[Job, bool]:
        """
        Returns (job, created). A repeated idempotency_key returns the
        existing job and created=False; no second row is ever written.
        """
        now = time.time()
        job_id = "job_" + uuid.uuid4().hex[:12]
        with self._tx() as conn:
            try:
                conn.execute(
                    "INSERT INTO jobs(job_id, idempotency_key, type, payload, status, attempts, "
                    "max_attempts, created_at, updated_at, available_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        job_id,
                        idempotency_key,
                        type,
                        json.dumps(payload),
                        JobStatus.QUEUED.value,
                        0,
                        max_attempts or self.retry_policy.max_attempts,
                        now,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                row = conn.execute(
                    "SELECT job_id FROM jobs WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
                return self.get(row["job_id"]), False
            self._event(conn, job_id, None, JobStatus.SUBMITTED.value, detail="api")
            self._event(conn, job_id, JobStatus.SUBMITTED.value, JobStatus.QUEUED.value)
        return self.get(job_id), True

    # ----------------------------------------------------------------- claim
    def claim(self, worker_id: str) -> Job | None:
        """
        Atomically take one runnable job: QUEUED/RETRYING with
        available_at <= now, or PROCESSING whose lease has expired
        (crashed worker).
        """
        now = time.time()
        lease_until = now + self.lease_seconds
        with self._tx() as conn:
            row = conn.execute(
                "SELECT job_id, status FROM jobs WHERE "
                "((status IN (?, ?) AND available_at <= ?) OR "
                " (status = ? AND lease_expires_at IS NOT NULL AND lease_expires_at < ?)) "
                "ORDER BY available_at ASC LIMIT 1",
                (
                    JobStatus.QUEUED.value,
                    JobStatus.RETRYING.value,
                    now,
                    JobStatus.PROCESSING.value,
                    now,
                ),
            ).fetchone()
            if row is None:
                return None
            prev = row["status"]
            reclaimed = prev == JobStatus.PROCESSING.value
            updated = conn.execute(
                "UPDATE jobs SET status=?, attempts=attempts+1, worker_id=?, "
                "lease_expires_at=?, updated_at=? WHERE job_id=? AND status=?",
                (
                    JobStatus.PROCESSING.value,
                    worker_id,
                    lease_until,
                    now,
                    row["job_id"],
                    prev,
                ),
            ).rowcount
            if updated != 1:
                return None
            self._event(
                conn,
                row["job_id"],
                prev,
                JobStatus.PROCESSING.value,
                worker_id,
                "lease_reclaimed_from_expired_worker" if reclaimed else "claimed",
            )
        return self.get(row["job_id"])

    def heartbeat(self, job_id: str, worker_id: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE jobs SET lease_expires_at=? WHERE job_id=? AND worker_id=? AND status=?",
                (time.time() + self.lease_seconds, job_id, worker_id, JobStatus.PROCESSING.value),
            )

    # -------------------------------------------------------------- complete
    def complete(self, job_id: str, worker_id: str, result: dict[str, Any] | None) -> None:
        now = time.time()
        with self._tx() as conn:
            n = conn.execute(
                "UPDATE jobs SET status=?, completed_at=?, updated_at=?, lease_expires_at=NULL, "
                "error=NULL, result=? WHERE job_id=? AND worker_id=? AND status=?",
                (
                    JobStatus.COMPLETED.value,
                    now,
                    now,
                    json.dumps(result or {}),
                    job_id,
                    worker_id,
                    JobStatus.PROCESSING.value,
                ),
            ).rowcount
            if n == 1:
                self._event(conn, job_id, JobStatus.PROCESSING.value, JobStatus.COMPLETED.value, worker_id)

    def fail(self, job_id: str, worker_id: str, error: str) -> JobStatus:
        """
        Record a failed attempt. Schedules a retry with backoff or moves
        the job to the DLQ when attempts are exhausted.
        """
        now = time.time()
        with self._tx() as conn:
            row = conn.execute(
                "SELECT attempts, max_attempts FROM jobs WHERE job_id=? AND worker_id=? AND status=?",
                (job_id, worker_id, JobStatus.PROCESSING.value),
            ).fetchone()
            if row is None:
                return JobStatus.FAILED
            if row["attempts"] >= row["max_attempts"]:
                conn.execute(
                    "UPDATE jobs SET status=?, updated_at=?, lease_expires_at=NULL, error=? WHERE job_id=?",
                    (JobStatus.DLQ.value, now, error[:4000], job_id),
                )
                self._event(conn, job_id, JobStatus.PROCESSING.value, JobStatus.FAILED.value, worker_id, error[:500])
                self._event(conn, job_id, JobStatus.FAILED.value, JobStatus.DLQ.value, worker_id, "max_attempts_exhausted")
                return JobStatus.DLQ
            delay = self.retry_policy.delay_for(row["attempts"])
            conn.execute(
                "UPDATE jobs SET status=?, updated_at=?, available_at=?, lease_expires_at=NULL, error=? "
                "WHERE job_id=?",
                (JobStatus.RETRYING.value, now, now + delay, error[:4000], job_id),
            )
            self._event(conn, job_id, JobStatus.PROCESSING.value, JobStatus.FAILED.value, worker_id, error[:500])
            self._event(conn, job_id, JobStatus.FAILED.value, JobStatus.RETRYING.value, worker_id, f"retry_in={delay:.2f}s")
            return JobStatus.RETRYING

    # ------------------------------------------------------------ side effect
    def side_effect_once(self, effect_key: str, job_id: str, detail: str = "") -> bool:
        """
        Returns True exactly once per effect_key across all workers and
        retries. The handler performs its external action only when this
        returns True.
        """
        with self._tx() as conn:
            try:
                conn.execute(
                    "INSERT INTO side_effects(effect_key, job_id, ts, detail) VALUES (?,?,?,?)",
                    (effect_key, job_id, time.time(), detail),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    # ------------------------------------------------------------------ read
    def get(self, job_id: str) -> Job:
        conn = self._conn()
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        events = [
            dict(e)
            for e in conn.execute(
                "SELECT ts, from_status, to_status, worker_id, detail FROM job_events "
                "WHERE job_id=? ORDER BY id",
                (job_id,),
            ).fetchall()
        ]
        return Job(
            job_id=row["job_id"],
            idempotency_key=row["idempotency_key"],
            type=row["type"],
            payload=json.loads(row["payload"]),
            status=JobStatus(row["status"]),
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            available_at=row["available_at"],
            lease_expires_at=row["lease_expires_at"],
            worker_id=row["worker_id"],
            completed_at=row["completed_at"],
            error=row["error"],
            result=json.loads(row["result"]) if row["result"] else None,
            events=events,
        )

    def list(self, status: JobStatus | None = None, limit: int = 100) -> list[Job]:
        conn = self._conn()
        if status:
            rows = conn.execute(
                "SELECT job_id FROM jobs WHERE status=? ORDER BY updated_at DESC LIMIT ?",
                (status.value, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT job_id FROM jobs ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self.get(r["job_id"]) for r in rows]

    def stats(self) -> dict[str, int]:
        conn = self._conn()
        counts = {s.value: 0 for s in JobStatus}
        for row in conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"):
            counts[row["status"]] = row["n"]
        counts["side_effects"] = conn.execute("SELECT COUNT(*) FROM side_effects").fetchone()[0]
        return counts

    def requeue_dlq(self, job_id: str) -> Job:
        now = time.time()
        with self._tx() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, attempts=0, available_at=?, updated_at=?, error=NULL "
                "WHERE job_id=? AND status=?",
                (JobStatus.QUEUED.value, now, now, job_id, JobStatus.DLQ.value),
            )
            self._event(conn, job_id, JobStatus.DLQ.value, JobStatus.QUEUED.value, None, "manual_requeue")
        return self.get(job_id)
