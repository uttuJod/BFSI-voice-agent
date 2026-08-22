from __future__ import annotations

import asyncio
from typing import Any

from rag import (
    RAGConfig,
    SelfCorrectingRAG,
)

from business import (
    BusinessToolExecutor,
    ToolExecutionResult,
)

from .guards import (
    apply_deterministic_guards,
)

from .router_schema import (
    IntegratedResponse,
    RouterDecision,
)

from .latency_metrics import LatencyTracker


class BFSIOrchestrator:

    def __init__(
        self,
        router,
        tool_executor: BusinessToolExecutor,
        rag_config: RAGConfig | None = None,
    ):

        self.router = router

        self.tools = tool_executor

        self.rag = SelfCorrectingRAG(
            rag_config
        )

        self.latency = LatencyTracker()

    # ============================================================
    # MAIN ENTRY
    # ============================================================

    async def handle(
        self,
        user_text: str,
        customer_id: str,
    ) -> IntegratedResponse:

        # ========================================================
        # 1. QWEN ROUTER
        # ========================================================

        try:

            with self.latency.measure("router"):
                raw_decision = (
                    await asyncio.to_thread(
                        self.router.route,
                        user_text,
                    )
                )

        except Exception as exc:

            fallback = (
                self._fallback_router()
            )

            return IntegratedResponse(
                user_text=user_text,
                router=fallback,

                final_response=(
                    "I couldn't safely process "
                    "that request."
                ),

                source="safe_fallback",

                guard_actions=[
                    (
                        "ROUTER_FAILURE:"
                        f"{type(exc).__name__}"
                    )
                ],
            )

        # ========================================================
        # 2. DETERMINISTIC GUARDS
        # ========================================================

        with self.latency.measure("guards"):
            decision, guard_actions = (
                apply_deterministic_guards(
                    user_text,
                    raw_decision,
                )
            )

        # ========================================================
        # 3. ROUTER CLARIFICATION
        # ========================================================

        if decision.needs_clarification:

            return IntegratedResponse(
                user_text=user_text,
                router=decision,

                final_response=(
                    decision.response.strip()
                    or (
                        "Could you clarify what "
                        "you need help with?"
                    )
                ),

                source="clarification",

                guard_actions=(
                    guard_actions
                ),
            )

        # ========================================================
        # 4. RAG
        # ========================================================

        rag_result = None

        if decision.requires_rag:

            with self.latency.measure("rag"):
                rag_result = (
                    await self.rag.answer(
                        user_text
                    )
                )

            if not rag_result.answerable:

                if (
                    rag_result
                    .clarification_question
                ):

                    final = (
                        rag_result
                        .clarification_question
                    )

                    source = (
                        "clarification"
                    )

                else:

                    final = (
                        "I don't have enough "
                        "verified policy information "
                        "to answer that safely."
                    )

                    source = (
                        "safe_fallback"
                    )

                return IntegratedResponse(
                    user_text=user_text,
                    router=decision,
                    final_response=final,
                    source=source,

                    rag_used=True,
                    rag_answerable=False,

                    rag_verdict=(
                        self._enum_value(
                            rag_result.verdict
                        )
                    ),

                    rag_citations=[],

                    guard_actions=(
                        guard_actions
                    ),
                )

        # ========================================================
        # 5. TOOL EXECUTION
        # ========================================================

        tool_result = None

        if (
            decision.requires_tool
            and decision.tool
        ):

            with self.latency.measure("tool"):
                tool_result = (
                    await asyncio.to_thread(
                        self.tools.execute,

                        decision.tool,

                        decision.arguments,

                        user_text,

                        customer_id,
                    )
                )

            # ----------------------------------------------------
            # TOOL NEEDS CLARIFICATION
            # ----------------------------------------------------

            if (
                tool_result
                .needs_clarification
            ):

                rag_answer = (
                    rag_result.answer
                    if (
                        rag_result
                        and rag_result.answerable
                    )
                    else None
                )

                final = (
                    self._combine(
                        rag_answer,
                        tool_result.user_message,
                    )
                )

                return self._response(
                    user_text=(
                        user_text
                    ),

                    decision=decision,

                    final_response=final,

                    source=(
                        "tool_clarification"
                    ),

                    rag_result=(
                        rag_result
                    ),

                    tool_result=(
                        tool_result
                    ),

                    guard_actions=(
                        guard_actions
                    ),
                )

            # ----------------------------------------------------
            # TOOL ERROR / BLOCK
            # ----------------------------------------------------

            if not tool_result.success:

                rag_answer = (
                    rag_result.answer
                    if (
                        rag_result
                        and rag_result.answerable
                    )
                    else None
                )

                final = (
                    self._combine(
                        rag_answer,
                        tool_result.user_message,
                    )
                )

                return self._response(
                    user_text=user_text,
                    decision=decision,
                    final_response=final,
                    source="tool_error",
                    rag_result=rag_result,
                    tool_result=tool_result,
                    guard_actions=(
                        guard_actions
                    ),
                )

        # ========================================================
        # 6. RAG + TOOL SUCCESS
        # ========================================================

        if (
            rag_result is not None
            and rag_result.answerable
            and tool_result is not None
            and tool_result.success
        ):

            final = self._combine(
                rag_result.answer,
                tool_result.user_message,
            )

            return self._response(
                user_text=user_text,
                decision=decision,
                final_response=final,
                source="rag_and_tool",
                rag_result=rag_result,
                tool_result=tool_result,
                guard_actions=(
                    guard_actions
                ),
            )

        # ========================================================
        # 7. RAG ONLY
        # ========================================================

        if (
            rag_result is not None
            and rag_result.answerable
        ):

            return self._response(
                user_text=user_text,
                decision=decision,

                final_response=(
                    rag_result.answer
                    or (
                        "I found relevant "
                        "policy information."
                    )
                ),

                source="rag",

                rag_result=rag_result,

                tool_result=None,

                guard_actions=(
                    guard_actions
                ),
            )

        # ========================================================
        # 8. TOOL ONLY
        # ========================================================

        if (
            tool_result is not None
            and tool_result.success
        ):

            return self._response(
                user_text=user_text,
                decision=decision,

                final_response=(
                    tool_result.user_message
                ),

                source="tool",

                rag_result=None,

                tool_result=(
                    tool_result
                ),

                guard_actions=(
                    guard_actions
                ),
            )

        # ========================================================
        # 9. ROUTER ONLY
        # ========================================================

        return IntegratedResponse(
            user_text=user_text,
            router=decision,

            final_response=(
                decision.response.strip()
                or (
                    "I couldn't determine a "
                    "safe response."
                )
            ),

            source="router",

            guard_actions=(
                guard_actions
            ),
        )

    # ============================================================
    # RESULT BUILDER
    # ============================================================

    def _response(
        self,
        user_text: str,
        decision: RouterDecision,
        final_response: str,
        source: str,
        rag_result,
        tool_result: ToolExecutionResult | None,
        guard_actions: list[str],
    ) -> IntegratedResponse:

        citations = (
            self._citations(
                rag_result
            )
            if rag_result
            else []
        )

        return IntegratedResponse(
            user_text=user_text,

            router=decision,

            final_response=(
                final_response
            ),

            source=source,

            rag_used=(
                rag_result is not None
            ),

            rag_answerable=(
                rag_result.answerable
                if rag_result
                else None
            ),

            rag_verdict=(
                self._enum_value(
                    rag_result.verdict
                )
                if rag_result
                else None
            ),

            rag_citations=(
                citations
            ),

            tool_executed=(
                tool_result
                is not None
            ),

            tool_success=(
                tool_result.success
                if tool_result
                else None
            ),

            tool_name=(
                decision.tool
                if tool_result
                else None
            ),

            tool_arguments=(
                decision.arguments
                if tool_result
                else {}
            ),

            tool_status=(
                self._enum_value(
                    tool_result.status
                )
                if tool_result
                else None
            ),

            tool_result=(
                tool_result.data
                if tool_result
                else {}
            ),

            tool_error_code=(
                tool_result.error_code
                if tool_result
                else None
            ),

            guard_actions=(
                guard_actions
            ),
        )

    # ============================================================
    # CITATIONS
    # ============================================================

    @staticmethod
    def _citations(
        rag_result,
    ) -> list[str]:

        citations: list[str] = []

        for citation in (
            rag_result.citations
            or []
        ):

            if isinstance(
                citation,
                str,
            ):
                value = citation

            elif hasattr(
                citation,
                "document_id",
            ):
                value = (
                    citation.document_id
                )

            elif isinstance(
                citation,
                dict,
            ):
                value = (
                    citation.get(
                        "document_id"
                    )
                    or citation.get(
                        "source"
                    )
                    or str(citation)
                )

            else:
                value = str(
                    citation
                )

            if (
                value
                and value
                not in citations
            ):
                citations.append(
                    value
                )

        return citations

    # ============================================================
    # SAFE COMBINATION
    # ============================================================

    @staticmethod
    def _combine(
        first: str | None,
        second: str | None,
    ) -> str:

        pieces = [
            item.strip()
            for item in (
                first,
                second,
            )
            if item
            and item.strip()
        ]

        return " ".join(
            pieces
        )

    # ============================================================
    # ENUM
    # ============================================================

    @staticmethod
    def _enum_value(
        value,
    ) -> str | None:

        if value is None:
            return None

        if hasattr(
            value,
            "value",
        ):
            return str(
                value.value
            )

        return str(
            value
        )

    # ============================================================
    # LATENCY METRICS
    # ============================================================

    def save_latency_metrics(
        self,
        path: str = "results/production_latency.json",
    ) -> dict[str, Any]:
        self.latency.save(path)
        return self.latency.summary()


    # ============================================================
    # FALLBACK
    # ============================================================

    @staticmethod
    def _fallback_router(
    ) -> RouterDecision:

        return RouterDecision(
            intent="human_escalation",
            confidence=0.0,
            requires_rag=False,
            requires_tool=False,
            tool=None,
            arguments={},
            response_style="apologetic",
            needs_clarification=False,
            response=(
                "I couldn't safely process "
                "that request."
            ),
        )