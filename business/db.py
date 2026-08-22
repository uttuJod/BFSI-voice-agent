from __future__ import annotations

import sqlite3
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,

    customer_id TEXT NOT NULL,

    account_status TEXT NOT NULL DEFAULT 'active',

    outstanding_balance REAL NOT NULL DEFAULT 0
        CHECK (outstanding_balance >= 0),

    currency TEXT NOT NULL DEFAULT 'INR',

    due_date TEXT,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS
idx_accounts_customer
ON accounts(customer_id);


CREATE TABLE IF NOT EXISTS payments (
    payment_id TEXT PRIMARY KEY,

    account_id TEXT NOT NULL,

    reported_amount REAL,

    payment_date TEXT,

    reference TEXT,

    status TEXT NOT NULL DEFAULT 'reported_unverified',

    source TEXT NOT NULL DEFAULT 'customer_report',

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (account_id)
        REFERENCES accounts(account_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS
idx_payments_account
ON payments(account_id);


CREATE TABLE IF NOT EXISTS promise_to_pay (
    promise_id TEXT PRIMARY KEY,

    account_id TEXT NOT NULL,

    promised_date TEXT NOT NULL,

    amount REAL,

    status TEXT NOT NULL DEFAULT 'recorded',

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (account_id)
        REFERENCES accounts(account_id)
        ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS callbacks (
    callback_id TEXT PRIMARY KEY,

    customer_id TEXT NOT NULL,

    requested_for TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'scheduled',

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
        ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS disputes (
    dispute_id TEXT PRIMARY KEY,

    account_id TEXT NOT NULL,

    reason TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'open',

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (account_id)
        REFERENCES accounts(account_id)
        ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS hardship_cases (
    hardship_id TEXT PRIMARY KEY,

    account_id TEXT NOT NULL,

    reason TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'open',

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (account_id)
        REFERENCES accounts(account_id)
        ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS human_escalations (
    escalation_id TEXT PRIMARY KEY,

    customer_id TEXT NOT NULL,

    reason TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'requested',

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
        ON DELETE CASCADE
);




CREATE TABLE IF NOT EXISTS customer_verification (
    customer_id TEXT PRIMARY KEY,

    method TEXT NOT NULL DEFAULT 'registered_mobile_last4',

    secret_salt TEXT NOT NULL,

    secret_hash TEXT NOT NULL,

    active INTEGER NOT NULL DEFAULT 1
        CHECK (active IN (0, 1)),

    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS call_history (
    call_id TEXT PRIMARY KEY,

    customer_id TEXT NOT NULL,

    event_type TEXT NOT NULL,

    summary TEXT NOT NULL,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
        ON DELETE CASCADE
);
"""


class BusinessDatabase:

    def __init__(
        self,
        path: str | Path,
    ):

        self.path = Path(
            path
        ).resolve()

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def connect(
        self,
    ) -> sqlite3.Connection:

        conn = sqlite3.connect(
            self.path,
            timeout=10.0,
        )

        conn.row_factory = (
            sqlite3.Row
        )

        conn.execute(
            "PRAGMA foreign_keys = ON"
        )

        conn.execute(
            "PRAGMA busy_timeout = 10000"
        )

        return conn

    @contextmanager
    def transaction(
        self,
    ) -> Iterator[sqlite3.Connection]:

        conn = self.connect()

        try:

            conn.execute(
                "BEGIN"
            )

            yield conn

            conn.commit()

        except Exception:

            conn.rollback()

            raise

        finally:

            conn.close()

    def initialize(
        self,
    ) -> None:

        with self.connect() as conn:

            conn.executescript(
                SCHEMA_SQL
            )

            conn.commit()

        self._enable_wal()

    def _enable_wal(
        self,
    ) -> None:

        with self.connect() as conn:

            conn.execute(
                "PRAGMA journal_mode = WAL"
            )

            conn.commit()

    def seed_demo_data(
        self,
    ) -> None:
        """
        Development-only demo customer.

        The inserts are idempotent.
        """

        with self.transaction() as conn:

            conn.execute(
                """
                INSERT OR IGNORE INTO customers (
                    customer_id,
                    name,
                    status
                )
                VALUES (?, ?, ?)
                """,
                (
                    "CUST-1001",
                    "Demo Customer",
                    "active",
                ),
            )

            conn.execute(
                """
                INSERT OR IGNORE INTO accounts (
                    account_id,
                    customer_id,
                    account_status,
                    outstanding_balance,
                    currency,
                    due_date
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "ACC-1001",
                    "CUST-1001",
                    "active",
                    12500.00,
                    "INR",
                    "2026-08-25",
                ),
            )

            #
            # DEVELOPMENT-ONLY identity verification factor.
            # The raw code is never stored. The demo code is 0641.
            #
            demo_code = "0641"
            demo_salt = "bfsi-demo-cust-1001-v1"
            demo_hash = hashlib.pbkdf2_hmac(
                "sha256",
                demo_code.encode("utf-8"),
                demo_salt.encode("utf-8"),
                200_000,
            ).hex()

            conn.execute(
                """
                INSERT OR IGNORE INTO customer_verification (
                    customer_id,
                    method,
                    secret_salt,
                    secret_hash,
                    active
                )
                VALUES (?, ?, ?, ?, 1)
                """,
                (
                    "CUST-1001",
                    "registered_mobile_last4",
                    demo_salt,
                    demo_hash,
                ),
            )

            conn.execute(
                """
                INSERT OR IGNORE INTO call_history (
                    call_id,
                    customer_id,
                    event_type,
                    summary
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    "CALL-DEMO-001",
                    "CUST-1001",
                    "account_created",
                    "Demo account initialized.",
                ),
            )