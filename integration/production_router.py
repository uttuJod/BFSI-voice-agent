from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from integration.vllm_router import VLLMBFSIRouter, SupportDecision, VLLMRouterError


@dataclass
class RouterCallMetrics:
    total_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    backend: str = "vllm"


class ProductionBFSIRouter:
    """
    Thin production wrapper around the benchmarked VLLMBFSIRouter.

    Important:
    - Does NOT change user text.
    - Does NOT modify predictions.
    - Does NOT add deterministic intent corrections.
    - Preserves the 118/120 benchmarked vLLM inference behavior.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8001/v1",
        model: str = "bfsi-router",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._router = VLLMBFSIRouter(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            max_tokens=320,
            use_structured_output=True,
        )

    def close(self) -> None:
        self._router.close()

    def health(self) -> dict[str, Any]:
        return self._router.health()

    def route(self, user_text: str) -> SupportDecision:
        return self._router.route(user_text)

    def route_with_metrics(
        self,
        user_text: str,
    ) -> tuple[SupportDecision, RouterCallMetrics]:
        decision, raw = self._router.route_with_metrics(user_text)

        metrics = RouterCallMetrics(
            total_ms=float(raw["total_ms"]),
            prompt_tokens=raw.get("prompt_tokens"),
            completion_tokens=raw.get("completion_tokens"),
            total_tokens=raw.get("total_tokens"),
            finish_reason=raw.get("finish_reason"),
        )
        return decision, metrics
