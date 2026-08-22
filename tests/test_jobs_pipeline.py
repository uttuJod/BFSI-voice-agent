"""
Guaranteed-delivery pipeline: unit-level proofs of the four required
failure behaviours plus throughput. Each test uses a fresh SQLite file.
The chaos scripts under tests/chaos/ run the same scenarios against real
separate worker processes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from jobs import HandlerRegistry, JobStatus, JobStore, RetryPolicy, Worker
from jobs import worker as worker_mod


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "SIDE_EFFECT_LOG", tmp_path / "effects.log")
    return JobStore(
        tmp_path / "jobs.db",
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=0.01, max_delay_s=0.05, jitter=0.0),
        lease_seconds=0.3,
    )


def effects(tmp_path) -> list[dict]:
    p = tmp_path / "effects.log"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def drain(store, worker, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not worker.run_once():
            s = store.stats()
            if s["QUEUED"] == 0 and s["RETRYING"] == 0 and s["PROCESSING"] == 0:
                return
            time.sleep(0.02)


def test_duplicate_submission_single_side_effect(store, tmp_path):
    j1, created1 = store.submit(idempotency_key="job-abc", type="send_email", payload={"to": "a@x"})
    j2, created2 = store.submit(idempotency_key="job-abc", type="send_email", payload={"to": "a@x"})
    assert created1 and not created2
    assert j1.job_id == j2.job_id
    w = Worker(store, worker_mod.DEFAULT_REGISTRY, worker_id="w1")
    drain(store, w)
    assert store.get(j1.job_id).status is JobStatus.COMPLETED
    assert len([e for e in effects(tmp_path) if e["kind"] == "email_sent"]) == 1
    assert store.stats()["COMPLETED"] == 1


def test_duplicate_delivery_after_retry_single_side_effect(store, tmp_path):
    # First attempt performs the side effect then fails AFTER it; retry
    # must not repeat the effect.
    reg = HandlerRegistry()
    calls = {"n": 0}

    @reg.register("flaky_after_effect")
    def handler(ctx):
        calls["n"] += 1
        if ctx.ledger.once("effect:" + ctx.job.idempotency_key):
            worker_mod._append_effect("external_call", {"job_id": ctx.job.job_id})
        if calls["n"] == 1:
            raise RuntimeError("crash after side effect")
        return {"ok": True}

    store.submit(idempotency_key="k1", type="flaky_after_effect", payload={})
    w = Worker(store, reg, worker_id="w1")
    drain(store, w)
    assert calls["n"] == 2
    assert len(effects(tmp_path)) == 1
    assert store.stats()["COMPLETED"] == 1


def test_retry_backoff_then_success(store):
    job, _ = store.submit(idempotency_key="k2", type="send_email", payload={"fail_times": 2})
    w = Worker(store, worker_mod.DEFAULT_REGISTRY, worker_id="w1")
    drain(store, w)
    final = store.get(job.job_id)
    assert final.status is JobStatus.COMPLETED
    assert final.attempts == 3
    transitions = [e["to_status"] for e in final.events]
    assert transitions.count("RETRYING") == 2
    assert transitions[-1] == "COMPLETED"


def test_permanent_failure_lands_in_dlq_with_error(store):
    job, _ = store.submit(idempotency_key="k3", type="sync_webhook", payload={"always_fail": True})
    w = Worker(store, worker_mod.DEFAULT_REGISTRY, worker_id="w1")
    drain(store, w)
    final = store.get(job.job_id)
    assert final.status is JobStatus.DLQ
    assert final.attempts == 3
    assert "injected permanent failure" in (final.error or "")
    assert [j.job_id for j in store.list(JobStatus.DLQ)] == [job.job_id]
    # Manual requeue path works and the job is runnable again.
    store.requeue_dlq(job.job_id)
    assert store.get(job.job_id).status is JobStatus.QUEUED


def test_worker_crash_mid_job_is_reclaimed_not_lost(store, tmp_path):
    job, _ = store.submit(idempotency_key="k4", type="generate_report", payload={})
    # Simulate a worker that claimed the job and died: claim, then never complete.
    claimed = store.claim("dead-worker")
    assert claimed is not None and claimed.status is JobStatus.PROCESSING
    assert store.claim("w2") is None  # lease still held
    time.sleep(0.35)  # lease expires
    w = Worker(store, worker_mod.DEFAULT_REGISTRY, worker_id="w2")
    assert w.run_once() is True
    final = store.get(job.job_id)
    assert final.status is JobStatus.COMPLETED
    assert final.attempts == 2
    assert any(e["detail"] == "lease_reclaimed_from_expired_worker" for e in final.events)
    assert len(effects(tmp_path)) == 1


def test_store_restart_preserves_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "SIDE_EFFECT_LOG", tmp_path / "effects.log")
    db = tmp_path / "jobs.db"
    s1 = JobStore(db, lease_seconds=1)
    for i in range(5):
        s1.submit(idempotency_key=f"r{i}", type="send_email", payload={"to": f"{i}@x"})
    assert s1.stats()["QUEUED"] == 5
    del s1  # "broker restart": drop every handle and reopen the file
    s2 = JobStore(db, lease_seconds=1)
    assert s2.stats()["QUEUED"] == 5
    w = Worker(s2, worker_mod.DEFAULT_REGISTRY, worker_id="w1")
    drain(s2, w)
    assert s2.stats()["COMPLETED"] == 5
    assert len(effects(tmp_path)) == 5


def test_throughput_is_measurable(store, tmp_path):
    n = 200
    for i in range(n):
        store.submit(idempotency_key=f"t{i}", type="process_document", payload={})
    w = Worker(store, worker_mod.DEFAULT_REGISTRY, worker_id="w1")
    t0 = time.perf_counter()
    drain(store, w, timeout=30)
    elapsed = time.perf_counter() - t0
    assert store.stats()["COMPLETED"] == n
    jobs_per_min = n / elapsed * 60
    (Path("results")).mkdir(exist_ok=True)
    Path("results/jobs_throughput.json").write_text(
        json.dumps({"jobs": n, "seconds": round(elapsed, 3), "jobs_per_minute": round(jobs_per_min)})
    )
    assert jobs_per_min > 500  # single worker, single process, SQLite WAL


def test_unknown_type_goes_to_dlq(store):
    job, _ = store.submit(idempotency_key="k5", type="nope", payload={})
    w = Worker(store, worker_mod.DEFAULT_REGISTRY, worker_id="w1")
    drain(store, w)
    assert store.get(job.job_id).status is JobStatus.DLQ


def test_api_contract(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from jobs import api

    monkeypatch.setenv("JOBS_DB", str(tmp_path / "api.db"))
    api._store = None
    c = TestClient(api.app)

    r = c.post("/v1/jobs", json={"type": "send_email", "payload": {"to": "u@x"}})
    assert r.status_code == 400  # missing Idempotency-Key

    r1 = c.post("/v1/jobs", json={"type": "send_email", "payload": {"to": "u@x"}}, headers={"Idempotency-Key": "job-abc"})
    r2 = c.post("/v1/jobs", json={"type": "send_email", "payload": {"to": "u@x"}}, headers={"Idempotency-Key": "job-abc"})
    assert r1.status_code == 202 and r1.json()["status"] == "QUEUED" and r1.json()["created"] is True
    assert r2.json()["job_id"] == r1.json()["job_id"] and r2.json()["created"] is False

    g = c.get(f"/v1/jobs/{r1.json()['job_id']}")
    assert g.status_code == 200
    body = g.json()
    assert set(["job_id", "type", "status", "attempts", "created_at", "completed_at", "error"]) <= set(body)
    assert c.get("/v1/jobs/stats").json()["QUEUED"] == 1
    assert c.get("/v1/jobs/missing").status_code == 404
