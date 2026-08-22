from __future__ import annotations

import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


def _find_attr(
    obj: Any,
    names: tuple[str, ...],
) -> Any | None:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _find_router(orchestrator: Any) -> Any:
    router = _find_attr(
        orchestrator,
        (
            "router",
            "_router",
            "llm_router",
            "_llm_router",
        ),
    )

    if router is None:
        raise RuntimeError(
            "Could not find router on BFSIOrchestrator."
        )

    return router


class SafeLatencyController:
    """
    Compatibility controller for both:
      - remote vLLM HTTP router
      - legacy local HF router

    For vLLM, there is no local model.generate() to patch or cancel.
    For a legacy local HF router, we only set safe inference flags.
    """

    def __init__(
        self,
        *,
        orchestrator: Any,
    ) -> None:
        self.orchestrator = orchestrator
        self.router = _find_router(orchestrator)
        self.model = _find_attr(
            self.router,
            ("model", "_model", "llm", "_llm"),
        )

    def install(self) -> None:
        # vLLM HTTP path: no local torch model exists.
        if self.model is None:
            setattr(
                self.orchestrator,
                "latency_controller",
                self,
            )
            logger.info(
                "vLLM latency mode installed | "
                "remote_router=True | model_patch=False"
            )
            return

        # Legacy HF compatibility path.
        try:
            self.model.eval()
        except Exception:
            pass

        try:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        except Exception:
            pass

        config = getattr(self.model, "config", None)
        if config is not None:
            try:
                config.use_cache = True
            except Exception:
                pass

        generation_config = getattr(
            self.model,
            "generation_config",
            None,
        )
        if generation_config is not None:
            try:
                generation_config.use_cache = True
            except Exception:
                pass

        setattr(
            self.orchestrator,
            "latency_controller",
            self,
        )

        logger.info(
            "Safe HF latency mode installed | "
            "use_cache=True | generate_monkey_patch=False"
        )

    def reset_cancel(self) -> None:
        # Compatibility with existing voice/runtime.py.
        return None


def _walk_objects(
    root: Any,
    *,
    max_depth: int = 4,
):
    queue = deque([(root, 0)])
    seen: set[int] = set()

    while queue:
        obj, depth = queue.popleft()

        object_id = id(obj)
        if object_id in seen:
            continue

        seen.add(object_id)
        yield obj

        if depth >= max_depth:
            continue

        try:
            values = vars(obj).values()
        except Exception:
            continue

        for value in values:
            if isinstance(
                value,
                (
                    str,
                    bytes,
                    int,
                    float,
                    bool,
                    type(None),
                ),
            ):
                continue

            if isinstance(
                value,
                (
                    dict,
                    list,
                    tuple,
                    set,
                ),
            ):
                iterable = (
                    value.values()
                    if isinstance(value, dict)
                    else value
                )
                for item in iterable:
                    queue.append((item, depth + 1))
                continue

            queue.append((value, depth + 1))


def move_rag_embeddings_to_cpu(
    orchestrator: Any,
) -> int:
    """
    Keep SentenceTransformer/MiniLM off the RTX GPU so the 8 GB GPU
    remains available for vLLM.
    """
    moved = 0

    for obj in _walk_objects(orchestrator):
        cls = obj.__class__

        module_name = str(
            getattr(cls, "__module__", "")
        ).lower()

        class_name = str(
            getattr(cls, "__name__", "")
        ).lower()

        if (
            "sentence_transformers" not in module_name
            and "sentencetransformer" not in class_name
        ):
            continue

        to_method = getattr(obj, "to", None)
        if not callable(to_method):
            continue

        try:
            to_method("cpu")
            moved += 1
        except Exception:
            logger.exception(
                "Could not move SentenceTransformer to CPU."
            )

    logger.info(
        "RAG embedding placement | "
        "sentence_transformers_moved_to_cpu=%d",
        moved,
    )

    return moved


def install_latency_optimizations(
    orchestrator: Any,
    *,
    max_new_tokens: int = 320,
    move_embeddings_cpu: bool = True,
) -> SafeLatencyController:
    """
    Public compatibility hook used by the existing runtime.

    IMPORTANT:
    - Does not change vLLM max_tokens.
    - Does not patch generation.
    - Does not change the Stage-2B prompt.
    - Does not change structured-output behavior.
    """
    controller = SafeLatencyController(
        orchestrator=orchestrator,
    )
    controller.install()

    if move_embeddings_cpu:
        move_rag_embeddings_to_cpu(orchestrator)

    return controller
