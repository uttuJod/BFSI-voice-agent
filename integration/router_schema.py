from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Intent = Literal[
    "payment_reminder",
    "promise_to_pay",
    "callback_request",
    "paid_already",
    "partial_payment",
    "financial_hardship",
    "dispute",
    "wrong_number",
    "refusal_to_pay",
    "human_escalation",
    "policy_question",
    "account_status",
    "identity_verification",
    "ambiguous",
    "out_of_scope",
    "privacy_sensitive",
    "prompt_injection",
]


ToolName = Literal[
    "get_customer_account",
    "get_outstanding_balance",
    "record_promise_to_pay",
    "schedule_callback",
    "record_payment_reported",
    "open_dispute",
    "record_financial_hardship",
    "request_human_escalation",
    "get_policy",
    "get_call_history",
]


ResponseStyle = Literal[
    "supportive",
    "neutral",
    "apologetic",
    "firm",
    "concise",
]


class RouterDecision(BaseModel):

    intent: Intent

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    requires_rag: bool

    requires_tool: bool

    tool: ToolName | None = None

    arguments: dict[str, Any] = Field(
        default_factory=dict
    )

    response_style: ResponseStyle

    needs_clarification: bool

    response: str


class IntegratedResponse(BaseModel):

    user_text: str

    router: RouterDecision

    final_response: str

    source: Literal[
        "router",
        "rag",
        "clarification",
        "safe_fallback",
        "tool",
        "rag_and_tool",
        "tool_clarification",
        "tool_error",
    ]

    # RAG
    rag_used: bool = False

    rag_answerable: bool | None = None

    rag_verdict: str | None = None

    rag_citations: list[str] = Field(
        default_factory=list
    )

    # TOOL
    tool_executed: bool = False

    tool_success: bool | None = None

    tool_name: str | None = None

    tool_arguments: dict[str, Any] = Field(
        default_factory=dict
    )

    tool_status: str | None = None

    tool_result: dict[str, Any] = Field(
        default_factory=dict
    )

    tool_error_code: str | None = None

    # GUARDS
    guard_actions: list[str] = Field(
        default_factory=list
    )