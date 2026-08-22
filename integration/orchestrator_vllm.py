from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Optional

from integration.production_router import ProductionBFSIRouter
from integration.latency_metrics import LatencyTracker


@dataclass
class OrchestratorResult:
    """
    Generic production result returned by BFSIOrchestrator.handle().
    """
    intent: str
    confidence: float
    requires_rag: bool
    requires_tool: bool
    tool: Optional[str]
    arguments: dict[str, Any]
    needs_clarification: bool
    response_style: str
    router_response: str
    rag_result: Any = None
    tool_result: Any = None
    final_response: Optional[str] = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class BFSIOrchestrator:
    """
    Production orchestrator wired to the benchmarked vLLM router.

    IMPORTANT:
    - Does NOT modify the router prompt.
    - Does NOT alter router predictions.
    - Does NOT add intent correction rules.
    - Does NOT normalize user text inside this class.
    - Uses the exact VLLMBFSIRouter configuration already benchmarked at 118/120.

    Supply your existing RAG and tool executor through:
        rag_handler(text, decision)
        tool_handler(tool_name, arguments, decision)

    Both handlers are optional, so the router can be tested independently.
    """

    def __init__(
        self,
        *,
        rag_handler: Optional[Callable[..., Any]] = None,
        tool_handler: Optional[Callable[..., Any]] = None,
        response_builder: Optional[Callable[..., str]] = None,
        vllm_base_url: str = "http://127.0.0.1:8001/v1",
        vllm_model: str = "bfsi-router",
        timeout_seconds: float = 30.0,
        latency_output_path: str = "results/production_latency.json",
    ) -> None:
        self.router = ProductionBFSIRouter(
            base_url=vllm_base_url,
            model=vllm_model,
            timeout_seconds=timeout_seconds,
        )

        self.rag_handler = rag_handler
        self.tool_handler = tool_handler
        self.response_builder = response_builder

        self.latency = LatencyTracker()
        self.latency_output_path = Path(latency_output_path)

    def close(self) -> None:
        self.router.close()

    def health(self) -> dict[str, Any]:
        return self.router.health()

    def _run_rag(self, user_text: str, decision: Any) -> Any:
        if self.rag_handler is None:
            return None

        with self.latency.measure("rag"):
            return self.rag_handler(user_text, decision)

    def _run_tool(self, decision: Any) -> Any:
        if self.tool_handler is None:
            return None

        if not decision.tool:
            return None

        with self.latency.measure("tool"):
            return self.tool_handler(
                decision.tool,
                decision.arguments,
                decision,
            )

    def _build_response(
        self,
        *,
        user_text: str,
        decision: Any,
        rag_result: Any,
        tool_result: Any,
    ) -> str:
        """
        If your project already has a grounded response generator,
        pass it as response_builder.

        Otherwise use the frozen router's own short response.
        """
        if self.response_builder is not None:
            with self.latency.measure("response_builder"):
                return self.response_builder(
                    user_text=user_text,
                    decision=decision,
                    rag_result=rag_result,
                    tool_result=tool_result,
                )

        return decision.response

    def handle(self, user_text: str) -> OrchestratorResult:
        """
        Main production entry point.

        Input should be the final text that your existing STT / deterministic
        pre-routing layer already produces.

        This class intentionally does NOT add another normalization layer.
        """
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be a non-empty string")

        user_text = user_text.strip()

        with self.latency.measure("orchestrator_total"):

            # ----------------------------------------------------------
            # 1. ROUTER
            # ----------------------------------------------------------
            with self.latency.measure("router"):
                decision, router_metrics = self.router.route_with_metrics(
                    user_text
                )

            # Record the vLLM-reported HTTP/inference duration separately.
            self.latency.add(
                "router_vllm_http",
                router_metrics.total_ms,
            )

            # ----------------------------------------------------------
            # 2. RAG
            # ----------------------------------------------------------
            rag_result = None

            if decision.requires_rag:
                rag_result = self._run_rag(
                    user_text,
                    decision,
                )

            # ----------------------------------------------------------
            # 3. BUSINESS TOOL
            # ----------------------------------------------------------
            tool_result = None

            if decision.requires_tool and decision.tool:
                tool_result = self._run_tool(decision)

            # ----------------------------------------------------------
            # 4. GROUNDED RESPONSE
            # ----------------------------------------------------------
            final_response = self._build_response(
                user_text=user_text,
                decision=decision,
                rag_result=rag_result,
                tool_result=tool_result,
            )

            result = OrchestratorResult(
                intent=decision.intent,
                confidence=decision.confidence,
                requires_rag=decision.requires_rag,
                requires_tool=decision.requires_tool,
                tool=decision.tool,
                arguments=dict(decision.arguments),
                needs_clarification=decision.needs_clarification,
                response_style=decision.response_style,
                router_response=decision.response,
                rag_result=rag_result,
                tool_result=tool_result,
                final_response=final_response,
            )

        return result

    def save_latency_metrics(
        self,
        path: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Save P50/P95/etc. for router, RAG, tools and full orchestrator.
        """
        output = Path(path) if path else self.latency_output_path

        self.latency.save(output)

        return self.latency.summary()


# ----------------------------------------------------------------------
# EXAMPLE ADAPTERS
# ----------------------------------------------------------------------
#
# Replace these with your existing project functions.
#
#
# def rag_handler(user_text, decision):
#     return rag.query(user_text)
#
#
# def tool_handler(tool_name, arguments, decision):
#     return execute_tool(tool_name, arguments)
#
#
# def response_builder(
#     *,
#     user_text,
#     decision,
#     rag_result,
#     tool_result,
# ):
#     # Plug in your existing grounded-response/localization layer here.
#     #
#     # Example priority:
#     #   1. clarification
#     #   2. tool-grounded response
#     #   3. RAG-grounded response
#     #   4. router response
#
#     if decision.needs_clarification:
#         return decision.response
#
#     if tool_result is not None:
#         return str(tool_result)
#
#     if rag_result is not None:
#         return str(rag_result)
#
#     return decision.response
#
#
# orchestrator = BFSIOrchestrator(
#     rag_handler=rag_handler,
#     tool_handler=tool_handler,
#     response_builder=response_builder,
# )
#
# result = orchestrator.handle("Mera outstanding balance kitna hai?")
# print(result.model_dump())
# print(orchestrator.save_latency_metrics())
