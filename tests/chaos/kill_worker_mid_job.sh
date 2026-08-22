#!/usr/bin/env bash
# Crash test: a worker is SIGKILLed while holding a job. A second worker
# must reclaim the job after the lease expires and complete it exactly once.
source "$(dirname "$0")/common.sh"
read -r JOB _ <<< "$(submit crash-1 generate_report '{"sleep_s": 5}')"
W1=$(start_worker w1); sleep 1.2
read -r st _ <<< "$(status "$JOB")"; [ "$st" = PROCESSING ] || fail "expected PROCESSING, got $st"
kill -9 "$W1"; echo "killed worker w1 (pid $W1) while job $JOB was PROCESSING"
W2=$(start_worker w2)
wait_status "$JOB" COMPLETED 20 || { kill "$W2"; fail "job not completed after reclaim"; }
kill "$W2"
read -r st attempts <<< "$(status "$JOB")"
[ "$attempts" = 2 ] || fail "expected 2 attempts, got $attempts"
[ "$(effects)" = 1 ] || fail "expected exactly 1 side effect, got $(effects)"
pass "worker killed mid-job, job reclaimed by w2, completed once (attempts=2, side_effects=1)"
