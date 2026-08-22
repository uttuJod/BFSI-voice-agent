#!/usr/bin/env bash
cd "$(dirname "$0")"
for s in duplicate_submit kill_worker_mid_job restart_store permanent_failure_dlq; do
  echo "== $s =="; bash "$s.sh" || exit 1
done
echo "ALL CHAOS TESTS PASSED"
