"""
REST API for the job pipeline.

    POST /v1/jobs            Idempotency-Key header required
    GET  /v1/jobs/stats
    GET  /v1/jobs/dlq
    GET  /v1/jobs/{job_id}
    POST /v1/jobs/{job_id}/requeue

Run:  uvicorn jobs.api:app --port 8010
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .store import JobStatus, JobStore

app = FastAPI(title="Guaranteed-delivery jobs")
_store: JobStore | None = None


def store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore(
            os.getenv("JOBS_DB", "results/jobs.db"),
            lease_seconds=float(os.getenv("JOBS_LEASE_S", "5")),
        )
    return _store


class SubmitJob(BaseModel):
    type: str = Field(..., examples=["send_email"])
    payload: dict[str, Any] = Field(default_factory=dict)
    max_attempts: int | None = Field(default=None, ge=1, le=20)


@app.post("/v1/jobs", status_code=202)
def submit(body: SubmitJob, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    if not idempotency_key:
        raise HTTPException(400, "Idempotency-Key header is required")
    job, created = store().submit(
        idempotency_key=idempotency_key,
        type=body.type,
        payload=body.payload,
        max_attempts=body.max_attempts,
    )
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "idempotency_key": job.idempotency_key,
        "created": created,
    }


@app.get("/v1/jobs/stats")
def stats():
    return store().stats()


@app.get("/v1/jobs/dlq")
def dlq():
    return [j.to_dict() for j in store().list(JobStatus.DLQ)]


@app.get("/v1/jobs/{job_id}")
def get(job_id: str):
    try:
        job = store().get(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    d = job.to_dict()
    d["events"] = job.events
    return d


@app.post("/v1/jobs/{job_id}/requeue")
def requeue(job_id: str):
    try:
        return store().requeue_dlq(job_id).to_dict()
    except KeyError:
        raise HTTPException(404, "job not found")
