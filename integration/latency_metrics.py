from __future__ import annotations

import json
import statistics
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class LatencyTracker:
    """
    Lightweight stage-level latency collector.

    Example:
        tracker = LatencyTracker()

        with tracker.measure("router"):
            decision = router.route(text)

        tracker.save("results/latency.json")
    """

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self._values[stage].append(elapsed_ms)

    def add(self, stage: str, elapsed_ms: float) -> None:
        with self._lock:
            self._values[stage].append(float(elapsed_ms))

    def last(self) -> dict[str, float]:
        """Most recent sample per stage (for per-turn records)."""
        with self._lock:
            return {
                stage: round(values[-1], 1)
                for stage, values in self._values.items()
                if values
            }

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile) - 1))
        return ordered[index]

    def summary(self) -> dict:
        with self._lock:
            data = {k: list(v) for k, v in self._values.items()}

        result = {}
        for stage, values in data.items():
            if not values:
                continue

            result[stage] = {
                "count": len(values),
                "min_ms": round(min(values), 1),
                "p50_ms": round(statistics.median(values), 1),
                "p95_ms": round(self._percentile(values, 0.95), 1),
                "max_ms": round(max(values), 1),
                "mean_ms": round(statistics.mean(values), 1),
            }

        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.summary(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
