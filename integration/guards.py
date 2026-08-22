from __future__ import annotations

import re

from .router_schema import RouterDecision


IDENTITY_PATTERNS = (
    r"\bverify me\b",
    r"\bverify myself\b",
    r"\bmy identity\b",
    r"\bidentity check\b",
    r"\bidentity verification\b",
    r"\bverification process\b",
    r"\bverification kaise\b",
    r"\bverification kya\b",
    r"\bverify kaise\b",
    r"\baccount.*verification\b",
    r"\bverification.*account\b",
    r"\bbefore.*account.*information\b",
    r"\bbefore.*showing.*balance\b",
    r"\bbefore.*accessing.*account\b",
)


THIRD_PARTY_TERMS = (
    "my mother",
    "my father",
    "my wife",
    "my husband",
    "my partner",
    "my friend",
    "my brother",
    "my sister",
    "someone else's",
    "someone else",
    "another person's",
    "another person",
    "third party",
)


PRIVATE_ACCOUNT_TERMS = (
    "balance",
    "owe",
    "outstanding",
    "account status",
    "payment history",
    "call history",
    "account details",
    "how much",
)


ACCOUNT_DATA_TOOLS = {
    "get_customer_account",
    "get_outstanding_balance",
    "get_call_history",
}


def _normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower().strip(),
    )


def _looks_like_own_identity_question(
    text: str,
) -> bool:

    q = _normalize(text)

    return any(
        re.search(pattern, q)
        for pattern in IDENTITY_PATTERNS
    )


def _looks_like_third_party_private_request(
    text: str,
) -> bool:

    q = _normalize(text)

    third_party = any(
        term in q
        for term in THIRD_PARTY_TERMS
    )

    private_fact = any(
        term in q
        for term in PRIVATE_ACCOUNT_TERMS
    )

    return (
        third_party
        and private_fact
    )


def apply_deterministic_guards(
    user_text: str,
    decision: RouterDecision,
) -> tuple[RouterDecision, list[str]]:
    """
    Apply only business-critical deterministic corrections.

    We intentionally do NOT rewrite every model decision.
    The frozen model already has strong routing performance.

    Returns:
        corrected decision
        list of guard actions
    """

    actions: list[str] = []

    data = decision.model_dump()

    # ============================================================
    # 1. THIRD-PARTY PRIVACY
    # ============================================================

    if _looks_like_third_party_private_request(
        user_text
    ):
        if (
            decision.intent
            != "privacy_sensitive"
            or decision.requires_tool
            or decision.tool is not None
        ):
            data["intent"] = (
                "privacy_sensitive"
            )

            data["requires_tool"] = False
            data["tool"] = None
            data["arguments"] = {}
            data["requires_rag"] = False
            data["needs_clarification"] = False

            data["response_style"] = "firm"

            data["response"] = (
                "I can't provide another person's "
                "private account information."
            )

            actions.append(
                "THIRD_PARTY_PRIVACY_GUARD"
            )

    # ============================================================
    # 2. OWN-ACCOUNT IDENTITY NORMALIZATION
    # ============================================================

    elif _looks_like_own_identity_question(
        user_text
    ):
        if (
            decision.intent
            in {
                "privacy_sensitive",
                "out_of_scope",
                "ambiguous",
            }
        ):
            data["intent"] = (
                "identity_verification"
            )

            # Match the frozen Stage-2B benchmark contract exactly:
            # own-account identity verification is routed through
            # get_customer_account with verification_required=True.
            data["requires_tool"] = True
            data["tool"] = "get_customer_account"
            data["arguments"] = {
                "verification_required": True
            }

            data["requires_rag"] = False
            data["needs_clarification"] = False
            data["response_style"] = "neutral"

            if not data.get("response"):
                data["response"] = (
                    "Sensitive account information requires "
                    "approved verification first."
                )

            actions.append(
                "IDENTITY_VERIFICATION_NORMALIZATION"
            )

    # ============================================================
    # 3. POLICY QUESTIONS MUST USE RAG
    # ============================================================

    if data["intent"] == "policy_question":

        if not data["requires_rag"]:
            actions.append(
                "POLICY_RAG_ENFORCED"
            )

        data["requires_rag"] = True

        # Pure policy questions should not execute business tools.
        if data["requires_tool"]:
            actions.append(
                "POLICY_TOOL_BLOCKED"
            )

        data["requires_tool"] = False
        data["tool"] = None
        data["arguments"] = {}

    # ============================================================
    # 4. PRIVACY / INJECTION MUST NEVER EXECUTE ACCOUNT TOOLS
    # ============================================================

    if data["intent"] in {
        "privacy_sensitive",
        "prompt_injection",
    }:

        if (
            data["requires_tool"]
            or data["tool"] is not None
        ):
            actions.append(
                "UNSAFE_TOOL_BLOCKED"
            )

        data["requires_tool"] = False
        data["tool"] = None
        data["arguments"] = {}

    # ============================================================
    # 5. TOOL CONSISTENCY
    # ============================================================

    if not data["requires_tool"]:
        data["tool"] = None

    if (
        data["requires_tool"]
        and data["tool"] is None
    ):
        # Never execute an unspecified tool.
        data["requires_tool"] = False

        actions.append(
            "MISSING_TOOL_BLOCKED"
        )

    # Additional privacy safety check for account-data tools.
    if (
        data["tool"]
        in ACCOUNT_DATA_TOOLS
        and _looks_like_third_party_private_request(
            user_text
        )
    ):
        data["intent"] = "privacy_sensitive"
        data["requires_tool"] = False
        data["tool"] = None
        data["arguments"] = {}

        actions.append(
            "ACCOUNT_DATA_TOOL_PRIVACY_BLOCK"
        )

    corrected = RouterDecision.model_validate(
        data
    )

    return corrected, actions