from __future__ import annotations

import re
import time
from dataclasses import dataclass


WRITE_TOOLS = {
    "record_promise_to_pay",
    "schedule_callback",
    "record_payment_reported",
    "open_dispute",
    "record_financial_hardship",
    "request_human_escalation",
}


@dataclass(frozen=True, slots=True)
class DuplicateHit:
    key: str
    grounded_response: str


class SessionWriteIdempotency:
    """
    Session-level protection against voice retries creating duplicate writes.

    It is intentionally conservative. A record is cached only AFTER a
    successful write tool execution.

    TTL keeps this scoped to immediate/retry duplicates, not legitimate
    future customer actions.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 60.0,
    ) -> None:
        self.ttl_seconds = max(
            5.0,
            float(ttl_seconds),
        )

        self._records: dict[
            str,
            tuple[float, str],
        ] = {}

    def _amount(
        self,
        text: str,
    ) -> str:
        match = re.search(
            r"(?:₹|rs\.?|inr)?\s*"
            r"(\d[\d,]*(?:\.\d{1,2})?)",
            text,
            re.IGNORECASE,
        )

        if not match:
            return "unspecified"

        return (
            match.group(1)
            .replace(",", "")
        )

    def _date_token(
        self,
        text: str,
    ) -> str:
        patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            (
                r"\b(?:january|february|march|april|may|june|"
                r"july|august|september|october|november|december)"
                r"\s+\d{1,2}(?:,\s*\d{4})?\b"
            ),
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                re.IGNORECASE,
            )

            if match:
                return (
                    " ".join(
                        match.group(0)
                        .lower()
                        .split()
                    )
                )

        return "unspecified"

    def candidate_key(
        self,
        text: str,
    ) -> str | None:
        t = " ".join(
            str(text or "")
            .lower()
            .split()
        )

        if not t:
            return None

        if re.search(
            r"\bwrong person\b|\bwrong number\b",
            t,
        ):
            return (
                "request_human_escalation:"
                "wrong_number"
            )

        if re.search(
            r"\bdispute\b|\binvestigate\b|\bformal review\b",
            t,
        ):
            return "open_dispute:payment"

        if re.search(
            r"\blost my job\b|\bjob loss\b|\bincome stopped\b|"
            r"\bsalary stopped\b|\bfinancial hardship\b|"
            r"\bcannot pay\b.*\b(?:job|income|medical|emergency)\b",
            t,
        ):
            return (
                "record_financial_hardship:"
                "hardship"
            )

        if re.search(
            r"\b(?:i paid|i have paid|already paid|payment made|"
            r"paid yesterday|friend paid for me)\b",
            t,
        ):
            return (
                "record_payment_reported:"
                + self._amount(t)
            )

        if re.search(
            r"\b(?:i will pay|i'll pay|promise to pay)\b",
            t,
        ):
            return (
                "record_promise_to_pay:"
                + self._date_token(t)
                + ":"
                + self._amount(t)
            )

        if re.search(
            r"\b(?:call|ring|phone|contact)\s+me\b",
            t,
        ):
            return (
                "schedule_callback:"
                + self._date_token(t)
            )

        return None

    def check(
        self,
        text: str,
    ) -> DuplicateHit | None:
        key = self.candidate_key(
            text
        )

        if key is None:
            return None

        record = self._records.get(
            key
        )

        if record is None:
            return None

        created_at, response = record

        if (
            time.monotonic()
            - created_at
            > self.ttl_seconds
        ):
            self._records.pop(
                key,
                None,
            )
            return None

        return DuplicateHit(
            key=key,
            grounded_response=response,
        )

    def remember_success(
        self,
        *,
        text: str,
        tool_name: str | None,
        grounded_response: str,
    ) -> None:
        if (
            tool_name
            not in WRITE_TOOLS
        ):
            return

        key = self.candidate_key(
            text
        )

        if key is None:
            key = (
                f"{tool_name}:"
                + " ".join(
                    text.lower().split()
                )
            )

        self._records[key] = (
            time.monotonic(),
            grounded_response,
        )
