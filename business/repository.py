from __future__ import annotations

import uuid
from typing import Any

from .db import BusinessDatabase


def _id(
    prefix: str,
) -> str:

    return (
        f"{prefix}-"
        f"{uuid.uuid4().hex[:12].upper()}"
    )


def _row_to_dict(
    row,
) -> dict[str, Any] | None:

    if row is None:
        return None

    return dict(row)


class BusinessRepository:

    def __init__(
        self,
        db: BusinessDatabase,
    ):

        self.db = db

    # ============================================================
    # CUSTOMER
    # ============================================================

    def customer_exists(
        self,
        customer_id: str,
    ) -> bool:

        with self.db.connect() as conn:

            row = conn.execute(
                """
                SELECT 1
                FROM customers
                WHERE customer_id = ?
                LIMIT 1
                """,
                (
                    customer_id,
                ),
            ).fetchone()

        return row is not None

    # ============================================================
    # ACCOUNTS
    # ============================================================

    def get_accounts(
        self,
        customer_id: str,
    ) -> list[dict[str, Any]]:

        with self.db.connect() as conn:

            rows = conn.execute(
                """
                SELECT
                    account_id,
                    customer_id,
                    account_status,
                    outstanding_balance,
                    currency,
                    due_date,
                    created_at,
                    updated_at
                FROM accounts
                WHERE customer_id = ?
                ORDER BY created_at
                """,
                (
                    customer_id,
                ),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def get_account(
        self,
        customer_id: str,
        account_id: str,
    ) -> dict[str, Any] | None:

        with self.db.connect() as conn:

            row = conn.execute(
                """
                SELECT
                    account_id,
                    customer_id,
                    account_status,
                    outstanding_balance,
                    currency,
                    due_date,
                    created_at,
                    updated_at
                FROM accounts
                WHERE customer_id = ?
                  AND account_id = ?
                LIMIT 1
                """,
                (
                    customer_id,
                    account_id,
                ),
            ).fetchone()

        return _row_to_dict(
            row
        )

    # ============================================================
    # PAYMENTS
    # ============================================================

    def record_payment_reported(
        self,
        account_id: str,
        amount: float | None,
        payment_date: str | None,
        reference: str | None,
    ) -> dict[str, Any]:

        payment_id = _id(
            "PAY"
        )

        with self.db.transaction() as conn:

            conn.execute(
                """
                INSERT INTO payments (
                    payment_id,
                    account_id,
                    reported_amount,
                    payment_date,
                    reference,
                    status,
                    source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payment_id,
                    account_id,
                    amount,
                    payment_date,
                    reference,
                    "reported_unverified",
                    "customer_report",
                ),
            )

            row = conn.execute(
                """
                SELECT *
                FROM payments
                WHERE payment_id = ?
                """,
                (
                    payment_id,
                ),
            ).fetchone()

        return dict(
            row
        )

    # ============================================================
    # PROMISE TO PAY
    # ============================================================

    def record_promise_to_pay(
        self,
        account_id: str,
        promised_date: str,
        amount: float | None,
    ) -> dict[str, Any]:

        promise_id = _id(
            "PTP"
        )

        with self.db.transaction() as conn:

            conn.execute(
                """
                INSERT INTO promise_to_pay (
                    promise_id,
                    account_id,
                    promised_date,
                    amount,
                    status
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    promise_id,
                    account_id,
                    promised_date,
                    amount,
                    "recorded",
                ),
            )

            row = conn.execute(
                """
                SELECT *
                FROM promise_to_pay
                WHERE promise_id = ?
                """,
                (
                    promise_id,
                ),
            ).fetchone()

        return dict(
            row
        )

    # ============================================================
    # CALLBACK
    # ============================================================

    def schedule_callback(
        self,
        customer_id: str,
        requested_for: str,
    ) -> dict[str, Any]:

        callback_id = _id(
            "CB"
        )

        with self.db.transaction() as conn:

            conn.execute(
                """
                INSERT INTO callbacks (
                    callback_id,
                    customer_id,
                    requested_for,
                    status
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    callback_id,
                    customer_id,
                    requested_for,
                    "scheduled",
                ),
            )

            row = conn.execute(
                """
                SELECT *
                FROM callbacks
                WHERE callback_id = ?
                """,
                (
                    callback_id,
                ),
            ).fetchone()

        return dict(
            row
        )

    # ============================================================
    # DISPUTE
    # ============================================================

    def open_dispute(
        self,
        account_id: str,
        reason: str,
    ) -> dict[str, Any]:

        dispute_id = _id(
            "DSP"
        )

        with self.db.transaction() as conn:

            conn.execute(
                """
                INSERT INTO disputes (
                    dispute_id,
                    account_id,
                    reason,
                    status
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    dispute_id,
                    account_id,
                    reason,
                    "open",
                ),
            )

            row = conn.execute(
                """
                SELECT *
                FROM disputes
                WHERE dispute_id = ?
                """,
                (
                    dispute_id,
                ),
            ).fetchone()

        return dict(
            row
        )

    # ============================================================
    # HARDSHIP
    # ============================================================

    def record_hardship(
        self,
        account_id: str,
        reason: str,
    ) -> dict[str, Any]:

        hardship_id = _id(
            "HARDSHIP"
        )

        with self.db.transaction() as conn:

            conn.execute(
                """
                INSERT INTO hardship_cases (
                    hardship_id,
                    account_id,
                    reason,
                    status
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    hardship_id,
                    account_id,
                    reason,
                    "open",
                ),
            )

            row = conn.execute(
                """
                SELECT *
                FROM hardship_cases
                WHERE hardship_id = ?
                """,
                (
                    hardship_id,
                ),
            ).fetchone()

        return dict(
            row
        )

    # ============================================================
    # HUMAN ESCALATION
    # ============================================================

    def request_human_escalation(
        self,
        customer_id: str,
        reason: str,
    ) -> dict[str, Any]:

        escalation_id = _id(
            "ESC"
        )

        with self.db.transaction() as conn:

            conn.execute(
                """
                INSERT INTO human_escalations (
                    escalation_id,
                    customer_id,
                    reason,
                    status
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    escalation_id,
                    customer_id,
                    reason,
                    "requested",
                ),
            )

            row = conn.execute(
                """
                SELECT *
                FROM human_escalations
                WHERE escalation_id = ?
                """,
                (
                    escalation_id,
                ),
            ).fetchone()

        return dict(
            row
        )

    # ============================================================
    # CALL HISTORY
    # ============================================================

    def add_call_event(
        self,
        customer_id: str,
        event_type: str,
        summary: str,
    ) -> dict[str, Any]:

        call_id = _id(
            "CALL"
        )

        with self.db.transaction() as conn:

            conn.execute(
                """
                INSERT INTO call_history (
                    call_id,
                    customer_id,
                    event_type,
                    summary
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    call_id,
                    customer_id,
                    event_type,
                    summary,
                ),
            )

            row = conn.execute(
                """
                SELECT *
                FROM call_history
                WHERE call_id = ?
                """,
                (
                    call_id,
                ),
            ).fetchone()

        return dict(
            row
        )

    def get_call_history(
        self,
        customer_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:

        limit = max(
            1,
            min(
                int(limit),
                50,
            ),
        )

        with self.db.connect() as conn:

            rows = conn.execute(
                """
                SELECT
                    call_id,
                    event_type,
                    summary,
                    created_at
                FROM call_history
                WHERE customer_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (
                    customer_id,
                    limit,
                ),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]