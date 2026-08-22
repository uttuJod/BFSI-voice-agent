"""
Barge-in stop-latency metrics.

Definition used for the "Barge-in stop latency" metric (problem
statement 3): time from the VAD detecting user speech onset while the
assistant is speaking, to the moment the browser has silenced playback.

    stop_latency = onset_to_cancel (server)
                 + one-way transport delay (half the ack round trip)
                 + client_stop (browser clearAudio duration)

Samples are accumulated per session and also appended to a JSONL file so
`bench/latency_report.py` can aggregate across sessions.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass(frozen=True)
class BargeInSample:
    ts: float
    onset_to_cancel_ms: float | None
    server_to_client_one_way_ms: float
    client_stop_ms: float
    stop_latency_ms: float


class BargeInMetrics:
    def __init__(self, *, jsonl_path: str | os.PathLike | None = None) -> None:
        self._samples: list[BargeInSample] = []
        self._rejected: list[dict] = []
        self._jsonl_path = Path(
            jsonl_path
            or os.getenv("BARGE_METRICS_JSONL", "results/barge_in_samples.jsonl")
        )

    def record(
        self,
        *,
        onset_to_cancel_ms: float | None,
        server_to_client_one_way_ms: float,
        client_stop_ms: float,
    ) -> BargeInSample:
        stop_latency_ms = (
            (onset_to_cancel_ms or 0.0)
            + max(server_to_client_one_way_ms, 0.0)
            + max(client_stop_ms, 0.0)
        )
        sample = BargeInSample(
            ts=time.time(),
            onset_to_cancel_ms=onset_to_cancel_ms,
            server_to_client_one_way_ms=server_to_client_one_way_ms,
            client_stop_ms=client_stop_ms,
            stop_latency_ms=stop_latency_ms,
        )
        self._samples.append(sample)
        self._append_jsonl(sample)
        return sample

    def _append_jsonl(self, sample: BargeInSample) -> None:
        try:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(sample)) + "\n")
        except OSError:
            # Metrics must never break the voice path.
            pass

    def record_rejected(self, *, reason: str, voiced_ms: float | None) -> None:
        """
        A VAD speech-start during assistant playback that did NOT become a
        confirmed barge-in. Counted as a suppressed false positive.
        """
        entry = {
            "ts": time.time(),
            "rejected": True,
            "reason": reason,
            "voiced_ms": round(voiced_ms, 1) if voiced_ms is not None else None,
        }
        self._rejected.append(entry)
        try:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    @property
    def rejected_count(self) -> int:
        return len(self._rejected)

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def p50(self) -> float:
        return _percentile([s.stop_latency_ms for s in self._samples], 0.5)

    @property
    def p95(self) -> float:
        return _percentile([s.stop_latency_ms for s in self._samples], 0.95)

    def summary(self) -> dict:
        values = [s.stop_latency_ms for s in self._samples]
        return {
            "count": len(values),
            "p50_ms": round(_percentile(values, 0.5), 1),
            "p95_ms": round(_percentile(values, 0.95), 1),
            "max_ms": round(max(values), 1) if values else 0.0,
            "target_ms": 500,
            "false_candidates_suppressed": len(self._rejected),
            "suppression_reasons": {
                r: sum(1 for e in self._rejected if e["reason"] == r)
                for r in sorted({e["reason"] for e in self._rejected})
            },
            "within_target_rate": (
                round(sum(v <= 500 for v in values) / len(values), 3)
                if values
                else None
            ),
        }
