#!/usr/bin/env bash
# Duplicate submission: same Idempotency-Key three times, one side effect.
source "$(dirname "$0")/common.sh"
read -r J1 C1 <<< "$(submit job-abc send_email '{"to":"user@example.com"}')"
read -r J2 C2 <<< "$(submit job-abc send_email '{"to":"user@example.com"}')"
read -r J3 C3 <<< "$(submit job-abc send_email '{"to":"user@example.com"}')"
[ "$J1" = "$J2" ] && [ "$J2" = "$J3" ] || fail "different job ids for the same key"
[ "$C1" = True ] && [ "$C2" = False ] && [ "$C3" = False ] || fail "created flags wrong: $C1 $C2 $C3"
W=$(start_worker w1); wait_status "$J1" COMPLETED 10 || { kill "$W"; fail "not completed"; }; kill "$W"
[ "$(effects)" = 1 ] || fail "expected 1 email, got $(effects)"
pass "3 submissions, 1 job, 1 email sent"
