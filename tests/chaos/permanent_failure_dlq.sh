#!/usr/bin/env bash
# Permanent failure: retries with backoff, then DLQ with the error recorded.
source "$(dirname "$0")/common.sh"
read -r JOB _ <<< "$(submit dlq-1 process_document '{"always_fail": true}')"
W=$(start_worker w1); wait_status "$JOB" DLQ 15 || { kill "$W"; fail "job never reached DLQ"; }; kill "$W"
python3 - "$JOB" <<'PY'
import os, sys; from jobs import JobStore
j = JobStore(os.environ["JOBS_DB"]).get(sys.argv[1])
assert j.attempts == 3, j.attempts
assert "injected permanent failure" in (j.error or "")
retries = [e for e in j.events if e["to_status"] == "RETRYING"]
assert len(retries) == 2, retries
print("attempts:", j.attempts, "| retries with backoff:", [e["detail"] for e in retries])
PY
[ "$(effects)" = 0 ] || fail "side effect recorded despite failure"
pass "3 attempts, 2 backoff retries, DLQ with error context, 0 side effects"
