from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta


@dataclass(frozen=True, slots=True)
class GuardedInput:
    original: str
    normalized: str
    actions: tuple[str, ...]


class PreOrchestratorGuard:
    """
    Deterministic normalization BEFORE the frozen Qwen router.

    Fixes only high-confidence edge cases:
    - wrong-person/self-identity wording
    - own-account "bank balance" wording
    - explicit month/day dates missing a year
    - callback "tomorrow at <numeric time>" to an exact calendar date

    It does not classify general intent and does not replace Qwen.
    """

    MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11,
        "december": 12,
    }

    MONTH_RE = re.compile(
        r"\b("
        + "|".join(MONTHS)
        + r")\s+(\d{1,2})(?:st|nd|rd|th)?"
        r"(?!\s*,?\s*\d{4})",
        re.IGNORECASE,
    )

    WRONG_PERSON_RE = re.compile(
        r"\b(?:i\s*am|i'm|im)\s+not\s+"
        r"[\w\u0900-\u097F][\w\u0900-\u097F .'-]{0,60}"
        r"(?:[.!?]|$)",
        re.IGNORECASE,
    )

    THIRD_PERSON_NUMBER_RE = re.compile(
        r"\b(?:wrong person|wrong number|"
        r"number belongs to someone else|"
        r"this is not the person you are looking for)\b",
        re.IGNORECASE,
    )

    OWN_BALANCE_RE = re.compile(
        r"\b(?:my|mine)\b.*\b(?:bank balance|account balance|"
        r"current balance|outstanding balance|amount due|balance due)\b",
        re.IGNORECASE,
    )

    THIRD_PARTY_RE = re.compile(
        r"\b(?:his|her|their|wife'?s|husband'?s|friend'?s|"
        r"mother'?s|father'?s|brother'?s|sister'?s|"
        r"someone else'?s|another customer'?s)\b",
        re.IGNORECASE,
    )

    CALLBACK_RE = re.compile(
        r"\b(?:call|ring|phone|contact)\s+me\b",
        re.IGNORECASE,
    )

    NUMERIC_TIME_RE = re.compile(
        r"\b(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        today_provider=None,
    ) -> None:
        self._today_provider = (
            today_provider
            or date.today
        )

    def normalize(
        self,
        text: str,
    ) -> GuardedInput:
        original = " ".join(
            str(text or "").split()
        ).strip()

        if not original:
            return GuardedInput(
                original="",
                normalized="",
                actions=(),
            )

        normalized = original
        actions: list[str] = []

        # High-confidence wrong-person wording.
        if (
            self.THIRD_PERSON_NUMBER_RE.search(normalized)
            or self.WRONG_PERSON_RE.search(normalized)
        ):
            normalized = "You have the wrong person."
            actions.append(
                "wrong_person_canonicalized"
            )

            return GuardedInput(
                original=original,
                normalized=normalized,
                actions=tuple(actions),
            )

        # First-person account balance should not become third-party privacy.
        if (
            self.OWN_BALANCE_RE.search(normalized)
            and not self.THIRD_PARTY_RE.search(normalized)
        ):
            normalized = (
                "What is my outstanding balance?"
            )
            actions.append(
                "own_balance_canonicalized"
            )

        today = self._today_provider()

        # "August 25" -> "August 25, 2026" (next valid occurrence).
        def replace_month_day(
            match: re.Match,
        ) -> str:
            month_name = match.group(1)
            day_num = int(match.group(2))
            month_num = self.MONTHS[
                month_name.lower()
            ]

            year = today.year

            try:
                candidate = date(
                    year,
                    month_num,
                    day_num,
                )
            except ValueError:
                return match.group(0)

            if candidate < today:
                year += 1

                try:
                    candidate = date(
                        year,
                        month_num,
                        day_num,
                    )
                except ValueError:
                    return match.group(0)

            actions.append(
                "missing_year_resolved"
            )

            return (
                f"{candidate.strftime('%B')} "
                f"{candidate.day}, "
                f"{candidate.year}"
            )

        normalized = self.MONTH_RE.sub(
            replace_month_day,
            normalized,
        )

        # A callback tomorrow AT a numeric time is an exact relative datetime.
        # We intentionally do NOT resolve "I will pay tomorrow" because the
        # current PTP policy requires explicit confirmation.
        if (
            self.CALLBACK_RE.search(normalized)
            and re.search(
                r"\btomorrow\b",
                normalized,
                re.IGNORECASE,
            )
            and self.NUMERIC_TIME_RE.search(normalized)
        ):
            tomorrow = (
                today
                + timedelta(days=1)
            )

            replacement = (
                f"{tomorrow.strftime('%B')} "
                f"{tomorrow.day}, "
                f"{tomorrow.year}"
            )

            normalized = re.sub(
                r"\btomorrow\b",
                replacement,
                normalized,
                flags=re.IGNORECASE,
            )

            actions.append(
                "callback_tomorrow_resolved"
            )

        return GuardedInput(
            original=original,
            normalized=normalized,
            actions=tuple(actions),
        )
