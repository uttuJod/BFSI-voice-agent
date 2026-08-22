from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from integration.production_router import ProductionBFSIRouter


CASES = [
    ("balance", "What is my outstanding balance?"),
    ("paid", "I already paid yesterday."),
    ("policy", "What is the grace period policy?"),
    ("privacy", "My mother wants me to ask what amount remains unpaid."),
    ("identity", "Please verify me before retrieving my account details."),
    ("vague_date", "I might manage ₹1900 sometime next week."),
    ("exact_date", "I will pay ₹1900 on 25 August 2026."),
    ("hinglish", "Mera outstanding balance kitna hai?"),
    ("hindi", "मैंने पहले ही भुगतान कर दिया है।"),
]


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * 0.95) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/production_router_smoke.json",
    )
    args = parser.parse_args()

    router = ProductionBFSIRouter()
    print("HEALTH:", router.health())

    rows = []
    times = []

    for name, text in CASES:
        started = time.perf_counter()
        decision, metrics = router.route_with_metrics(text)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        times.append(elapsed_ms)

        row = {
            "name": name,
            "text": text,
            "decision": decision.model_dump(),
            "elapsed_ms": round(elapsed_ms, 1),
            "vllm_ms": metrics.total_ms,
        }
        rows.append(row)

        print(
            f"{name:12s} "
            f"{decision.intent:24s} "
            f"{elapsed_ms:7.1f} ms"
        )

    summary = {
        "n": len(rows),
        "p50_ms": round(statistics.median(times), 1),
        "p95_ms": round(p95(times), 1),
        "mean_ms": round(statistics.mean(times), 1),
        "rows": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:", out)


if __name__ == "__main__":
    main()
