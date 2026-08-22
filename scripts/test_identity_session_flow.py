from __future__ import annotations

from pathlib import Path

from business import (
    BusinessDatabase,
    BusinessRepository,
    BusinessToolExecutor,
    VerificationRegistry,
)


def main() -> None:
    db_path = Path(
        "results/identity_session_test.sqlite3"
    )

    if db_path.exists():
        db_path.unlink()

    db = BusinessDatabase(
        db_path
    )
    db.initialize()
    db.seed_demo_data()

    repo = BusinessRepository(
        db
    )
    registry = VerificationRegistry(
        db,
        verified_ttl_seconds=600,
        max_attempts=3,
    )
    tools = BusinessToolExecutor(
        repo,
        verification_registry=registry,
    )

    session_id = registry.create_session(
        "CUST-1001"
    )

    with registry.bind(
        session_id
    ):
        blocked = tools.execute(
            "get_outstanding_balance",
            {},
            "What is my outstanding balance?",
            "CUST-1001",
        )

    print("1. UNVERIFIED ACCOUNT ACCESS")
    print("status:", blocked.status)
    print("error_code:", blocked.error_code)
    print("data:", blocked.data)
    assert (
        blocked.error_code
        == "IDENTITY_VERIFICATION_REQUIRED"
    )
    assert blocked.data == {}

    print()
    print("2. START CHALLENGE")
    print(
        registry.begin(
            session_id,
            "CUST-1001",
            pending_request=(
                "What is my outstanding balance?"
            ),
        )
    )

    wrong = registry.consume_input(
        session_id,
        "CUST-1001",
        "1234",
    )

    print()
    print("3. WRONG CODE")
    print(wrong)
    assert wrong.handled
    assert not wrong.success

    correct = registry.consume_input(
        session_id,
        "CUST-1001",
        "0641",
    )

    print()
    print("4. CORRECT CODE")
    print(correct)
    assert correct.success
    assert (
        correct.resume_text
        == "What is my outstanding balance?"
    )

    with registry.bind(
        session_id
    ):
        allowed = tools.execute(
            "get_outstanding_balance",
            {},
            "What is my outstanding balance?",
            "CUST-1001",
        )

    print()
    print("5. VERIFIED ACCOUNT ACCESS")
    print("status:", allowed.status)
    print("success:", allowed.success)
    print("data:", allowed.data)
    assert allowed.success
    assert (
        allowed.data[
            "outstanding_balance"
        ]
        == 12500.0
    )

    registry.close_session(
        session_id
    )

    print()
    print("IDENTITY SESSION FLOW: PASS")


if __name__ == "__main__":
    main()
