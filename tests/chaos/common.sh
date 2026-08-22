#!/usr/bin/env bash
# Shared helpers for the chaos scripts. Each script is self-contained:
# it uses its own SQLite file and side-effect log under results/chaos/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export JOBS_DB="results/chaos/$(basename "$0" .sh).db"
export JOB_SIDE_EFFECT_LOG="results/chaos/$(basename "$0" .sh).effects.log"
export JOBS_LEASE_S="${JOBS_LEASE_S:-1}"
mkdir -p results/chaos
rm -f "$JOBS_DB" "$JOBS_DB-wal" "$JOBS_DB-shm" "$JOB_SIDE_EFFECT_LOG"

submit() {  # submit <idempotency_key> <type> <payload_json>
  python3 - "$1" "$2" "$3" <<'PY'
import os, sys, json
from jobs import JobStore
s = JobStore(os.environ["JOBS_DB"], lease_seconds=float(os.environ["JOBS_LEASE_S"]))
job, created = s.submit(idempotency_key=sys.argv[1], type=sys.argv[2], payload=json.loads(sys.argv[3]), max_attempts=3)
print(job.job_id, created)
PY
}
status() {  # status <job_id>
  python3 - "$1" <<'PY'
import os, sys
from jobs import JobStore
s = JobStore(os.environ["JOBS_DB"])
j = s.get(sys.argv[1]); print(j.status.value, j.attempts)
PY
}
effects() { [ -f "$JOB_SIDE_EFFECT_LOG" ] && wc -l < "$JOB_SIDE_EFFECT_LOG" || echo 0; }
start_worker() { python3 -m jobs.worker --worker-id "$1" --lease-seconds "$JOBS_LEASE_S" > "results/chaos/$1.log" 2>&1 & echo $!; }
wait_status() {  # wait_status <job_id> <STATUS> <timeout_s>
  for _ in $(seq 1 $(( $3 * 10 ))); do
    read -r st _ <<< "$(status "$1")"; [ "$st" = "$2" ] && return 0; sleep 0.1
  done; return 1
}
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; exit 1; }
