#!/usr/bin/env bash
# Broker restart: jobs are queued, every process holding the store is
# stopped, the store is reopened, and all jobs still complete.
source "$(dirname "$0")/common.sh"
for i in 1 2 3 4 5; do submit "restart-$i" sync_webhook '{"url":"https://example.com/hook","sleep_s":0.6}' > /dev/null; done
W=$(start_worker w1); sleep 0.3; kill -9 "$W"; echo "stopped every process using the store"
python3 - <<'PY'
import os; from jobs import JobStore
print("queue after restart:", JobStore(os.environ["JOBS_DB"]).stats())
PY
W=$(start_worker w2); sleep 6; kill "$W"
done_count=$(python3 -c 'import os; from jobs import JobStore; print(JobStore(os.environ["JOBS_DB"]).stats()["COMPLETED"])')
[ "$done_count" = 5 ] || fail "expected 5 completed, got $done_count"
[ "$(effects)" = 5 ] || fail "expected 5 webhooks, got $(effects)"
pass "5 jobs survived store restart and completed once each"
