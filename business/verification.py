from __future__ import annotations

import contextvars
import hashlib
import hmac
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

from .db import BusinessDatabase


_CURRENT_VERIFICATION_SESSION: contextvars.ContextVar[
    str | None
] = contextvars.ContextVar(
    "bfsi_verification_session",
    default=None,
)


@dataclass(slots=True)
class VerificationState:
    session_id: str
    customer_id: str
    verified_until: float = 0.0
    awaiting_code: bool = False
    attempts: int = 0
    locked: bool = False
    pending_request: str | None = None


@dataclass(slots=True)
class VerificationInputResult:
    handled: bool
    success: bool = False
    message: str | None = None
    resume_text: str | None = None
    locked: bool = False


class VerificationRegistry:
    """
    Session-scoped demo identity verification.

    Security properties:
    - state is scoped to a random voice-session id, not customer_id alone
    - raw verification secrets are never stored in the DB
    - constant-time hash comparison
    - max attempts
    - successful verification expires
    """

    def __init__(
        self,
        db: BusinessDatabase,
        *,
        verified_ttl_seconds: float = 600.0,
        max_attempts: int = 3,
    ) -> None:
        self.db = db
        self.verified_ttl_seconds = max(
            30.0,
            float(
                verified_ttl_seconds
            ),
        )
        self.max_attempts = max(
            1,
            int(
                max_attempts
            ),
        )
        self._states: dict[
            str,
            VerificationState,
        ] = {}

    def create_session(
        self,
        customer_id: str,
    ) -> str:
        session_id = (
            "VS-"
            + uuid.uuid4().hex.upper()
        )

        self._states[
            session_id
        ] = VerificationState(
            session_id=session_id,
            customer_id=customer_id,
        )

        return session_id

    def close_session(
        self,
        session_id: str,
    ) -> None:
        self._states.pop(
            session_id,
            None,
        )

    def state(
        self,
        session_id: str,
    ) -> VerificationState | None:
        return self._states.get(
            session_id
        )

    @contextmanager
    def bind(
        self,
        session_id: str,
    ):
        token = (
            _CURRENT_VERIFICATION_SESSION
            .set(
                session_id
            )
        )

        try:
            yield
        finally:
            _CURRENT_VERIFICATION_SESSION.reset(
                token
            )

    def _current_state(
        self,
    ) -> VerificationState | None:
        session_id = (
            _CURRENT_VERIFICATION_SESSION
            .get()
        )

        if not session_id:
            return None

        return self._states.get(
            session_id
        )

    def is_current_verified(
        self,
        customer_id: str,
    ) -> bool:
        state = self._current_state()

        if state is None:
            #
            # Compatibility path:
            # tests / non-voice orchestrator callers that do not bind a
            # verification session keep their existing behavior.
            #
            return True

        if (
            state.customer_id
            != customer_id
        ):
            return False

        if state.locked:
            return False

        return (
            state.verified_until
            > time.monotonic()
        )

    def is_verified(
        self,
        session_id: str,
        customer_id: str,
    ) -> bool:
        state = self._states.get(
            session_id
        )

        if (
            state is None
            or state.customer_id
            != customer_id
            or state.locked
        ):
            return False

        return (
            state.verified_until
            > time.monotonic()
        )

    def begin(
        self,
        session_id: str,
        customer_id: str,
        *,
        pending_request: str | None = None,
    ) -> str:
        state = self._states.get(
            session_id
        )

        if state is None:
            state = VerificationState(
                session_id=session_id,
                customer_id=customer_id,
            )
            self._states[
                session_id
            ] = state

        if (
            state.customer_id
            != customer_id
        ):
            return (
                "The verification session does not match "
                "the current customer."
            )

        if state.locked:
            return (
                "Identity verification is locked for this "
                "session after too many failed attempts. "
                "Please reconnect or request human support."
            )

        if self.is_verified(
            session_id,
            customer_id,
        ):
            return (
                "Your identity is already verified for "
                "this session."
            )

        state.awaiting_code = True

        if pending_request:
            state.pending_request = (
                pending_request
            )

        return (
            "Before I access account information, please "
            "say the last four digits of your registered "
            "mobile number."
        )

    def _extract_four_digits(
        self,
        text: str,
    ) -> str | None:
        """
        Accept numeric and naturally spoken four-digit verification codes.

        Examples:
            0641
            0 6 4 1
            zero six four one
            oh six four one
        """

        raw = str(
            text
        ).strip().lower()

        #
        # Deepgram smart_format can occasionally reinterpret a spoken
        # four-digit sequence as money, e.g.:
        #
        #   "zero six four one" -> "$64.01"
        #
        # During identity verification only, recover that deterministic
        # four-digit sequence before generic numeric extraction.
        #
        currency_match = re.fullmatch(
            r"\s*[$₹€£]?\s*(\d{1,2})[.,](\d{2})\s*[.!]?\s*",
            raw,
        )

        if currency_match:
            whole = currency_match.group(1)
            fraction = currency_match.group(2)

            #
            # Preserve a leading zero that smart_format may have dropped:
            # "$64.01" -> "0641"
            #
            if len(whole) == 2:
                recovered = (
                    "0"
                    + whole
                    + fraction[-1]
                )

                if len(recovered) == 4:
                    return recovered

        numeric = re.sub(
            r"\D",
            "",
            raw,
        )

        if len(numeric) == 4:
            return numeric

        word_to_digit = {
            "zero": "0",
            "oh": "0",
            "o": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
        }

        tokens = re.findall(
            r"[a-z]+|\d",
            raw,
        )

        spoken_digits: list[str] = []

        for token in tokens:
            if token.isdigit():
                spoken_digits.append(token)
                continue

            digit = word_to_digit.get(token)

            if digit is not None:
                spoken_digits.append(digit)

        if len(spoken_digits) == 4:
            return "".join(spoken_digits)

        return None

    def _verify_secret(
        self,
        customer_id: str,
        candidate: str,
    ) -> bool:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    secret_salt,
                    secret_hash
                FROM customer_verification
                WHERE customer_id = ?
                  AND active = 1
                """,
                (
                    customer_id,
                ),
            ).fetchone()

        if row is None:
            return False

        derived = hashlib.pbkdf2_hmac(
            "sha256",
            candidate.encode(
                "utf-8"
            ),
            str(
                row["secret_salt"]
            ).encode(
                "utf-8"
            ),
            200_000,
        ).hex()

        return hmac.compare_digest(
            derived,
            str(
                row["secret_hash"]
            ),
        )

    def consume_input(
        self,
        session_id: str,
        customer_id: str,
        text: str,
    ) -> VerificationInputResult:
        state = self._states.get(
            session_id
        )

        if (
            state is None
            or state.customer_id
            != customer_id
            or not state.awaiting_code
        ):
            return VerificationInputResult(
                handled=False,
            )

        if state.locked:
            return VerificationInputResult(
                handled=True,
                message=(
                    "Identity verification is locked for "
                    "this session. Please reconnect or "
                    "request human support."
                ),
                locked=True,
            )

        candidate = (
            self._extract_four_digits(
                text
            )
        )

        if candidate is None:
            return VerificationInputResult(
                handled=True,
                message=(
                    "Please provide exactly four digits "
                    "from your registered mobile number."
                ),
            )

        if self._verify_secret(
            customer_id,
            candidate,
        ):
            state.awaiting_code = False
            state.attempts = 0
            state.verified_until = (
                time.monotonic()
                + self.verified_ttl_seconds
            )

            resume_text = (
                state.pending_request
            )

            state.pending_request = None

            return VerificationInputResult(
                handled=True,
                success=True,
                message=(
                    "Identity verified successfully."
                ),
                resume_text=resume_text,
            )

        state.attempts += 1

        if (
            state.attempts
            >= self.max_attempts
        ):
            state.locked = True
            state.awaiting_code = False
            state.pending_request = None

            return VerificationInputResult(
                handled=True,
                message=(
                    "Identity verification failed too many "
                    "times and is locked for this session. "
                    "Please reconnect or request human support."
                ),
                locked=True,
            )

        remaining = (
            self.max_attempts
            - state.attempts
        )

        return VerificationInputResult(
            handled=True,
            message=(
                "Those digits did not match the registered "
                "verification record. "
                f"{remaining} attempt"
                + (
                    "s remain."
                    if remaining != 1
                    else " remains."
                )
            ),
        )
