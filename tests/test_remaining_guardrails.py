from datetime import date

from voice.pre_orchestrator import (
    PreOrchestratorGuard,
)
from voice.idempotency import (
    SessionWriteIdempotency,
)


def fixed_today():
    return date(2026, 8, 20)


def test_wrong_person_is_canonicalized():
    guard = PreOrchestratorGuard(
        today_provider=fixed_today
    )

    out = guard.normalize(
        "I'm not Utkarsh."
    )

    assert (
        out.normalized
        == "You have the wrong person."
    )


def test_august_25_gets_current_future_year():
    guard = PreOrchestratorGuard(
        today_provider=fixed_today
    )

    out = guard.normalize(
        "I will pay on August 25."
    )

    assert (
        out.normalized
        == "I will pay on August 25, 2026."
    )


def test_past_month_day_rolls_to_next_year():
    guard = PreOrchestratorGuard(
        today_provider=fixed_today
    )

    out = guard.normalize(
        "I will pay on August 10."
    )

    assert (
        out.normalized
        == "I will pay on August 10, 2027."
    )


def test_callback_tomorrow_with_time_is_resolved():
    guard = PreOrchestratorGuard(
        today_provider=fixed_today
    )

    out = guard.normalize(
        "Call me tomorrow at 3 PM."
    )

    assert (
        out.normalized
        == "Call me August 21, 2026 at 3 PM."
    )


def test_own_bank_balance_canonicalizes():
    guard = PreOrchestratorGuard(
        today_provider=fixed_today
    )

    out = guard.normalize(
        "What is my current bank balance?"
    )

    assert (
        out.normalized
        == "What is my outstanding balance?"
    )


def test_write_idempotency_hits_only_after_success():
    guard = SessionWriteIdempotency(
        ttl_seconds=60
    )

    text = "I paid 5000."

    assert guard.check(text) is None

    guard.remember_success(
        text=text,
        tool_name="record_payment_reported",
        grounded_response="recorded",
    )

    assert guard.check(text) is not None


def test_read_only_query_not_idempotency_blocked():
    guard = SessionWriteIdempotency(
        ttl_seconds=60
    )

    assert (
        guard.candidate_key(
            "What is my grace period?"
        )
        is None
    )
