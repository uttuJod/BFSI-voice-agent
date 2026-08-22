from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from .repository import BusinessRepository
from .verification import VerificationRegistry
from .schemas import ToolExecutionResult


ACCOUNT_TOOLS = {
    "get_customer_account",
    "get_outstanding_balance",
    "record_promise_to_pay",
    "record_payment_reported",
    "open_dispute",
    "record_financial_hardship",
}


VERIFICATION_PROTECTED_TOOLS = {
    "get_customer_account",
    "get_outstanding_balance",
    "record_promise_to_pay",
    "schedule_callback",
    "record_payment_reported",
    "open_dispute",
    "record_financial_hardship",
    "get_call_history",
}


SUPPORTED_TOOLS = {
    "get_customer_account",
    "get_outstanding_balance",
    "record_promise_to_pay",
    "schedule_callback",
    "record_payment_reported",
    "open_dispute",
    "record_financial_hardship",
    "request_human_escalation",
    "get_call_history",
    "get_policy",
}


VAGUE_DATE_TERMS = (
    "tomorrow",
    "next week",
    "later",
    "after salary",
    "after payday",
    "month end",
    "month-end",
    "sometime",
    "soon",
)


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


class BusinessToolExecutor:

    def __init__(
        self,
        repository: BusinessRepository,
        verification_registry: VerificationRegistry | None = None,
    ):

        self.repo = repository
        self.verification_registry = verification_registry

    # ============================================================
    # PUBLIC EXECUTION ENTRY POINT
    # ============================================================

    def execute(
        self,
        tool: str,
        arguments: dict[str, Any] | None,
        user_text: str,
        customer_id: str,
    ) -> ToolExecutionResult:

        arguments = dict(
            arguments or {}
        )

        if tool not in SUPPORTED_TOOLS:

            return ToolExecutionResult.error(
                tool=tool,
                message=(
                    "That business action is not "
                    "available."
                ),
                error_code="UNSUPPORTED_TOOL",
            )

        if not customer_id:

            return ToolExecutionResult.blocked(
                tool=tool,
                message=(
                    "I cannot access account actions "
                    "without an authenticated customer."
                ),
                error_code="NO_SESSION_CUSTOMER",
            )

        if not self.repo.customer_exists(
            customer_id
        ):

            return ToolExecutionResult.blocked(
                tool=tool,
                message=(
                    "The current customer session "
                    "could not be verified."
                ),
                error_code="CUSTOMER_NOT_FOUND",
            )

        #
        # Voice production sessions are verification-gated.
        #
        # The registry is optional so isolated business/orchestrator
        # regression tests retain their existing direct-call behavior.
        #
        if (
            self.verification_registry is not None
            and tool in VERIFICATION_PROTECTED_TOOLS
            and not self.verification_registry
            .is_current_verified(
                customer_id
            )
        ):
            return ToolExecutionResult.clarify(
                tool=tool,
                message=(
                    "Identity verification is required "
                    "before this account action."
                ),
                error_code=(
                    "IDENTITY_VERIFICATION_REQUIRED"
                ),
            )

        #
        # Never trust the model for identity/session ownership.
        #
        arguments.pop(
            "customer_id",
            None,
        )

        handlers = {
            "get_customer_account":
                self._get_customer_account,

            "get_outstanding_balance":
                self._get_outstanding_balance,

            "record_promise_to_pay":
                self._record_promise_to_pay,

            "schedule_callback":
                self._schedule_callback,

            "record_payment_reported":
                self._record_payment_reported,

            "open_dispute":
                self._open_dispute,

            "record_financial_hardship":
                self._record_financial_hardship,

            "request_human_escalation":
                self._request_human_escalation,

            "get_call_history":
                self._get_call_history,

            "get_policy":
                self._get_policy,
        }

        try:

            return handlers[
                tool
            ](
                arguments=arguments,
                user_text=user_text,
                customer_id=customer_id,
            )

        except Exception as exc:

            return ToolExecutionResult.error(
                tool=tool,
                message=(
                    "The requested account action "
                    "could not be completed safely."
                ),
                error_code=(
                    f"{type(exc).__name__}"
                ),
            )

    # ============================================================
    # ACCOUNT RESOLUTION
    # ============================================================

    def _resolve_account(
        self,
        customer_id: str,
        arguments: dict[str, Any],
        tool: str,
    ) -> tuple[
        dict[str, Any] | None,
        ToolExecutionResult | None,
    ]:

        requested_account_id = (
            arguments.get(
                "account_id"
            )
        )

        if requested_account_id:

            account = (
                self.repo.get_account(
                    customer_id=customer_id,
                    account_id=str(
                        requested_account_id
                    ),
                )
            )

            if account is None:

                return (
                    None,
                    ToolExecutionResult.blocked(
                        tool=tool,
                        message=(
                            "That account does not belong "
                            "to the current authenticated "
                            "customer session."
                        ),
                        error_code=(
                            "ACCOUNT_OWNERSHIP_MISMATCH"
                        ),
                    ),
                )

            return account, None

        accounts = (
            self.repo.get_accounts(
                customer_id
            )
        )

        if not accounts:

            return (
                None,
                ToolExecutionResult.error(
                    tool=tool,
                    message=(
                        "No account was found for "
                        "the current customer."
                    ),
                    error_code=(
                        "NO_ACCOUNT"
                    ),
                ),
            )

        if len(accounts) > 1:

            return (
                None,
                ToolExecutionResult.clarify(
                    tool=tool,
                    message=(
                        "You have more than one account. "
                        "Which account should I use?"
                    ),
                    error_code=(
                        "MULTIPLE_ACCOUNTS"
                    ),
                ),
            )

        return accounts[0], None

    # ============================================================
    # GET CUSTOMER ACCOUNT
    # ============================================================

    def _get_customer_account(
        self,
        arguments,
        user_text,
        customer_id,
    ) -> ToolExecutionResult:

        # --------------------------------------------------------
        # IDENTITY VERIFICATION GATE
        #
        # A valid application session/customer_id is NOT equivalent
        # to completed identity verification for sensitive account
        # disclosure.
        #
        # The frozen router marks verification requests with:
        #     {"verification_required": True}
        #
        # Until a real approved verification service exists, never
        # return account details for that path. Ask the caller to
        # complete verification instead.
        # --------------------------------------------------------
        if bool(
            arguments.get(
                "verification_required",
                False,
            )
        ):
            if (
                self.verification_registry is None
                or not self.verification_registry
                .is_current_verified(
                    customer_id
                )
            ):
                return ToolExecutionResult.clarify(
                    tool="get_customer_account",
                    message=(
                        "Please complete the approved identity "
                        "verification step before I access or disclose "
                        "account details."
                    ),
                    error_code=(
                        "IDENTITY_VERIFICATION_REQUIRED"
                    ),
                )

        account, error = (
            self._resolve_account(
                customer_id,
                arguments,
                "get_customer_account",
            )
        )

        if error:
            return error

        safe_account = {
            "account_id":
                account["account_id"],

            "account_status":
                account["account_status"],

            "currency":
                account["currency"],

            "due_date":
                account["due_date"],
        }

        return ToolExecutionResult.ok(
            tool="get_customer_account",
            message=(
                f"The account status is "
                f"{account['account_status']}."
            ),
            data=safe_account,
        )

    # ============================================================
    # BALANCE
    # ============================================================

    def _get_outstanding_balance(
        self,
        arguments,
        user_text,
        customer_id,
    ) -> ToolExecutionResult:

        account, error = (
            self._resolve_account(
                customer_id,
                arguments,
                "get_outstanding_balance",
            )
        )

        if error:
            return error

        balance = float(
            account[
                "outstanding_balance"
            ]
        )

        currency = (
            account["currency"]
        )

        return ToolExecutionResult.ok(
            tool="get_outstanding_balance",
            message=(
                f"Your current outstanding "
                f"balance is {balance:.2f} "
                f"{currency}."
            ),
            data={
                "account_id":
                    account["account_id"],

                "outstanding_balance":
                    balance,

                "currency":
                    currency,
            },
        )

    # ============================================================
    # PAYMENT REPORTED
    # ============================================================

    def _record_payment_reported(
        self,
        arguments,
        user_text,
        customer_id,
    ) -> ToolExecutionResult:

        account, error = (
            self._resolve_account(
                customer_id,
                arguments,
                "record_payment_reported",
            )
        )

        if error:
            return error

        amount = (
            self._safe_float(
                arguments.get(
                    "amount"
                )
            )
        )

        payment_date = (
            arguments.get(
                "payment_date"
            )
            or arguments.get(
                "date"
            )
        )

        reference = (
            arguments.get(
                "reference"
            )
            or arguments.get(
                "reference_number"
            )
        )

        row = (
            self.repo.record_payment_reported(
                account_id=(
                    account["account_id"]
                ),
                amount=amount,
                payment_date=(
                    str(payment_date)
                    if payment_date
                    else None
                ),
                reference=(
                    str(reference)
                    if reference
                    else None
                ),
            )
        )

        self.repo.add_call_event(
            customer_id=customer_id,
            event_type=(
                "payment_reported"
            ),
            summary=(
                "Customer reported that "
                "a payment was made."
            ),
        )

        return ToolExecutionResult.ok(
            tool="record_payment_reported",
            message=(
                "I recorded your reported payment "
                "for verification. It has not been "
                "treated as confirmed yet."
            ),
            data={
                "payment_id":
                    row["payment_id"],

                "status":
                    row["status"],

                "account_id":
                    row["account_id"],
            },
        )

    # ============================================================
    # DISPUTE
    # ============================================================

    def _open_dispute(
        self,
        arguments,
        user_text,
        customer_id,
    ) -> ToolExecutionResult:

        account, error = (
            self._resolve_account(
                customer_id,
                arguments,
                "open_dispute",
            )
        )

        if error:
            return error

        reason = str(
            arguments.get(
                "reason"
            )
            or "customer_requested_investigation"
        ).strip()

        row = self.repo.open_dispute(
            account_id=(
                account["account_id"]
            ),
            reason=reason,
        )

        self.repo.add_call_event(
            customer_id=customer_id,
            event_type="dispute_opened",
            summary=(
                f"Dispute {row['dispute_id']} "
                f"opened."
            ),
        )

        return ToolExecutionResult.ok(
            tool="open_dispute",
            message=(
                "I opened a dispute for the "
                "payment-status investigation."
            ),
            data={
                "dispute_id":
                    row["dispute_id"],

                "status":
                    row["status"],

                "account_id":
                    row["account_id"],
            },
        )

    # ============================================================
    # HARDSHIP
    # ============================================================

    def _record_financial_hardship(
        self,
        arguments,
        user_text,
        customer_id,
    ) -> ToolExecutionResult:

        account, error = (
            self._resolve_account(
                customer_id,
                arguments,
                "record_financial_hardship",
            )
        )

        if error:
            return error

        reason = str(
            arguments.get(
                "reason"
            )
            or "financial_difficulty"
        ).strip()

        row = (
            self.repo.record_hardship(
                account_id=(
                    account["account_id"]
                ),
                reason=reason,
            )
        )

        self.repo.add_call_event(
            customer_id=customer_id,
            event_type=(
                "hardship_recorded"
            ),
            summary=(
                f"Financial hardship case "
                f"{row['hardship_id']} recorded."
            ),
        )

        return ToolExecutionResult.ok(
            tool="record_financial_hardship",
            message=(
                "I recorded the financial-hardship "
                "request for review. No specific "
                "hardship outcome has been approved "
                "yet."
            ),
            data={
                "hardship_id":
                    row["hardship_id"],

                "status":
                    row["status"],

                "reason":
                    row["reason"],
            },
        )

    # ============================================================
    # HUMAN ESCALATION
    # ============================================================

    def _request_human_escalation(
        self,
        arguments,
        user_text,
        customer_id,
    ) -> ToolExecutionResult:

        reason = str(
            arguments.get(
                "reason"
            )
            or "customer_requested_human"
        ).strip()

        row = (
            self.repo.request_human_escalation(
                customer_id=customer_id,
                reason=reason,
            )
        )

        self.repo.add_call_event(
            customer_id=customer_id,
            event_type=(
                "human_escalation_requested"
            ),
            summary=(
                f"Human escalation "
                f"{row['escalation_id']} "
                f"requested."
            ),
        )

        return ToolExecutionResult.ok(
            tool="request_human_escalation",
            message=(
                "I requested review by a human "
                "support specialist."
            ),
            data={
                "escalation_id":
                    row["escalation_id"],

                "status":
                    row["status"],
            },
        )

    # ============================================================
    # PROMISE TO PAY
    # ============================================================

    def _record_promise_to_pay(
        self,
        arguments,
        user_text,
        customer_id,
    ) -> ToolExecutionResult:

        account, error = (
            self._resolve_account(
                customer_id,
                arguments,
                "record_promise_to_pay",
            )
        )

        if error:
            return error

        model_date = (
            arguments.get(
                "promised_date"
            )
            or arguments.get(
                "date"
            )
        )

        if not model_date:

            return ToolExecutionResult.clarify(
                tool="record_promise_to_pay",
                message=(
                    "What exact calendar date "
                    "would you like to promise "
                    "the payment for?"
                ),
                error_code=(
                    "MISSING_PROMISE_DATE"
                ),
            )

        validated_date = (
            self._validate_explicit_date(
                user_text=user_text,
                model_date=str(
                    model_date
                ),
            )
        )

        if validated_date is None:

            return ToolExecutionResult.clarify(
                tool="record_promise_to_pay",
                message=(
                    "Please provide an exact "
                    "calendar date for the "
                    "promise to pay."
                ),
                error_code=(
                    "UNVERIFIED_PROMISE_DATE"
                ),
            )

        amount = (
            self._safe_float(
                arguments.get(
                    "amount"
                )
            )
        )

        row = (
            self.repo.record_promise_to_pay(
                account_id=(
                    account["account_id"]
                ),
                promised_date=(
                    validated_date
                ),
                amount=amount,
            )
        )

        self.repo.add_call_event(
            customer_id=customer_id,
            event_type=(
                "promise_to_pay_recorded"
            ),
            summary=(
                "Promise to pay recorded for "
                f"{validated_date}."
            ),
        )

        return ToolExecutionResult.ok(
            tool="record_promise_to_pay",
            message=(
                "Your promise to pay has been "
                f"recorded for {validated_date}."
            ),
            data={
                "promise_id":
                    row["promise_id"],

                "promised_date":
                    row["promised_date"],

                "amount":
                    row["amount"],

                "status":
                    row["status"],
            },
        )

    # ============================================================
    # CALLBACK
    # ============================================================

    def _schedule_callback(
        self,
        arguments,
        user_text,
        customer_id,
    ) -> ToolExecutionResult:

        requested_for = (
            arguments.get(
                "scheduled_for"
            )
            or arguments.get(
                "requested_for"
            )
            or arguments.get(
                "datetime"
            )
        )

        if not requested_for:

            return ToolExecutionResult.clarify(
                tool="schedule_callback",
                message=(
                    "What exact date and time "
                    "would you prefer for the "
                    "callback?"
                ),
                error_code=(
                    "MISSING_CALLBACK_TIME"
                ),
            )

        normalized = (
            self._validate_datetime(
                str(
                    requested_for
                )
            )
        )

        if normalized is None:

            return ToolExecutionResult.clarify(
                tool="schedule_callback",
                message=(
                    "Please provide an exact "
                    "callback date and time."
                ),
                error_code=(
                    "INVALID_CALLBACK_TIME"
                ),
            )

        row = (
            self.repo.schedule_callback(
                customer_id=customer_id,
                requested_for=normalized,
            )
        )

        self.repo.add_call_event(
            customer_id=customer_id,
            event_type=(
                "callback_scheduled"
            ),
            summary=(
                "Callback scheduled for "
                f"{normalized}."
            ),
        )

        return ToolExecutionResult.ok(
            tool="schedule_callback",
            message=(
                "The callback request has "
                f"been scheduled for {normalized}."
            ),
            data={
                "callback_id":
                    row["callback_id"],

                "requested_for":
                    row["requested_for"],

                "status":
                    row["status"],
            },
        )

    # ============================================================
    # CALL HISTORY
    # ============================================================

    def _get_call_history(
        self,
        arguments,
        user_text,
        customer_id,
    ) -> ToolExecutionResult:

        limit = (
            arguments.get(
                "limit",
                10,
            )
        )

        try:
            limit = int(
                limit
            )
        except Exception:
            limit = 10

        history = (
            self.repo.get_call_history(
                customer_id=customer_id,
                limit=limit,
            )
        )

        if not history:

            return ToolExecutionResult.ok(
                tool="get_call_history",
                message=(
                    "There is no recent call "
                    "history available."
                ),
                data={
                    "history": [],
                },
            )

        return ToolExecutionResult.ok(
            tool="get_call_history",
            message=(
                f"I found {len(history)} recent "
                f"support-history entries."
            ),
            data={
                "history":
                    history,
            },
        )

    # ============================================================
    # POLICY
    # ============================================================

    def _get_policy(
        self,
        arguments,
        user_text,
        customer_id,
    ) -> ToolExecutionResult:

        #
        # Important architectural boundary:
        # policy comes from SelfCorrectingRAG.
        #
        return ToolExecutionResult.blocked(
            tool="get_policy",
            message=(
                "Policy information must be "
                "retrieved through the grounded "
                "policy knowledge system."
            ),
            error_code=(
                "USE_RAG_FOR_POLICY"
            ),
        )

    # ============================================================
    # DATE GUARDS
    # ============================================================

    @classmethod
    def _validate_explicit_date(
        cls,
        user_text: str,
        model_date: str,
    ) -> str | None:

        q = user_text.lower()

        if any(
            vague in q
            for vague in VAGUE_DATE_TERMS
        ):
            return None

        model_normalized = (
            cls._parse_date_string(
                model_date
            )
        )

        if model_normalized is None:
            return None

        explicit_dates = (
            cls._extract_dates(
                user_text
            )
        )

        #
        # Critical:
        # model is not allowed to invent a calendar date.
        #
        if not explicit_dates:
            return None

        if (
            model_normalized
            not in explicit_dates
        ):
            return None

        return model_normalized

    @classmethod
    def _extract_dates(
        cls,
        text: str,
    ) -> set[str]:

        found: set[str] = set()

        # --------------------------------------------------------
        # ISO
        # --------------------------------------------------------
        for match in re.findall(
            r"\b\d{4}-\d{1,2}-\d{1,2}\b",
            text,
            flags=re.IGNORECASE,
        ):
            parsed = cls._parse_date_string(
                match
            )

            if parsed:
                found.add(
                    parsed
                )

        # --------------------------------------------------------
        # Slash dates
        #
        # Deepgram smart_format may emit:
        #     08/25/2026
        #
        # Existing user text may also contain:
        #     25/08/2026
        #
        # We do not guess ambiguous dates such as 08/09/2026.
        # Instead, both valid interpretations are returned and the
        # router-normalized ISO date must match one of them.
        # --------------------------------------------------------
        for match in re.findall(
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            text,
            flags=re.IGNORECASE,
        ):
            first, second, year = (
                int(part)
                for part in match.split("/")
            )

            candidates: set[str] = set()

            # DD/MM/YYYY candidate
            try:
                candidates.add(
                    date(
                        year,
                        second,
                        first,
                    ).isoformat()
                )
            except ValueError:
                pass

            # MM/DD/YYYY candidate
            try:
                candidates.add(
                    date(
                        year,
                        first,
                        second,
                    ).isoformat()
                )
            except ValueError:
                pass

            found.update(
                candidates
            )

        # --------------------------------------------------------
        # Hyphen dates (DD-MM-YYYY only, preserved behavior)
        # --------------------------------------------------------
        for match in re.findall(
            r"\b\d{1,2}-\d{1,2}-\d{4}\b",
            text,
            flags=re.IGNORECASE,
        ):
            parsed = cls._parse_date_string(
                match
            )

            if parsed:
                found.add(
                    parsed
                )

        # --------------------------------------------------------
        # Named-month dates
        # --------------------------------------------------------
        named_patterns = [
            (
                r"\b\d{1,2}\s+"
                r"(?:january|february|march|april|may|june|"
                r"july|august|september|october|november|december)"
                r"\s+\d{4}\b"
            ),
            (
                r"\b(?:january|february|march|april|may|june|"
                r"july|august|september|october|november|december)"
                r"\s+\d{1,2},?\s+\d{4}\b"
            ),
        ]

        for pattern in named_patterns:

            for match in re.findall(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):

                parsed = (
                    cls._parse_date_string(
                        match
                    )
                )

                if parsed:
                    found.add(
                        parsed
                    )

        return found

    @staticmethod
    def _parse_date_string(
        value: str,
    ) -> str | None:

        value = (
            value.strip()
            .replace(",", "")
        )

        formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%d %B %Y",
            "%B %d %Y",
        ]

        for fmt in formats:

            try:

                parsed = datetime.strptime(
                    value,
                    fmt,
                ).date()

                return (
                    parsed.isoformat()
                )

            except ValueError:
                continue

        return None

    @staticmethod
    def _validate_datetime(
        value: str,
    ) -> str | None:

        value = value.strip()

        formats = [
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]

        for fmt in formats:

            try:

                parsed = (
                    datetime.strptime(
                        value,
                        fmt,
                    )
                )

                return parsed.strftime(
                    "%Y-%m-%dT%H:%M"
                )

            except ValueError:
                continue

        return None

    # ============================================================
    # NUMERIC
    # ============================================================

    @staticmethod
    def _safe_float(
        value,
    ) -> float | None:

        if value is None:
            return None

        try:

            number = float(
                value
            )

            if number < 0:
                return None

            return number

        except (
            TypeError,
            ValueError,
        ):
            return None