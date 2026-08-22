from __future__ import annotations

from business import (
    BusinessDatabase,
    BusinessRepository,
    BusinessToolExecutor,
)


def main() -> None:
    db = BusinessDatabase(
        "results/business_executor_regression.sqlite3"
    )
    db.initialize()
    db.seed_demo_data()

    repo = BusinessRepository(db)
    tools = BusinessToolExecutor(repo)

    # ------------------------------------------------------------
    # 1. Deepgram-style MM/DD/YYYY should verify against router ISO
    # ------------------------------------------------------------
    result = tools.execute(
        "record_promise_to_pay",
        {
            "date": "2026-08-25",
            "amount": 900,
        },
        "I will pay 900 rupees on 08/25/2026.",
        "CUST-1001",
    )

    print("DATE TEST")
    print("status:", result.status)
    print("success:", result.success)
    print("message:", result.user_message)
    print("data:", result.data)
    print()

    # ------------------------------------------------------------
    # 2. Identity verification must NOT disclose account details
    # ------------------------------------------------------------
    result = tools.execute(
        "get_customer_account",
        {
            "verification_required": True,
        },
        "Please verify me before showing my account details.",
        "CUST-1001",
    )

    print("IDENTITY TEST")
    print("status:", result.status)
    print("success:", result.success)
    print("needs_clarification:", result.needs_clarification)
    print("error_code:", result.error_code)
    print("message:", result.user_message)
    print("data:", result.data)


if __name__ == "__main__":
    main()
