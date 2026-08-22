"""
Per-turn voice latency records.

Every completed voice turn appends one JSON line to
`results/voice_turn_samples.jsonl` (override with VOICE_METRICS_JSONL).
`python -m bench.latency_report` aggregates these into the per-stage
P50/P95 table required by the low-latency multilingual problem
statement.

Stage names are stable and documented in docs/EVALUATION.md.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class TurnMetrics:
    def __init__(self, *, jsonl_path: str | os.PathLike | None = None) -> None:
        self._jsonl_path = Path(
            jsonl_path
            or os.getenv("VOICE_METRICS_JSONL", "results/voice_turn_samples.jsonl")
        )
        self.count = 0

    def record(
        self,
        *,
        generation_id: int,
        output_language: str,
        intent: str | None,
        rag_used: bool,
        tool: str | None,
        stages: dict[str, Any],
    ) -> dict[str, Any]:
        record = {
            "ts": time.time(),
            "generation_id": generation_id,
            "output_language": output_language,
            "intent": intent,
            "rag_used": rag_used,
            "tool": tool,
            "stages": {
                k: (round(float(v), 1) if isinstance(v, (int, float)) else v)
                for k, v in stages.items()
            },
        }
        self.count += 1
        try:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return record
