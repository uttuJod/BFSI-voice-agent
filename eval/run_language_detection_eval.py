"""
Language-detection accuracy on eval/datasets/language_detection_eval.json.

Reports:
  * detection accuracy (english / hindi / mixed collapsed to output language)
  * response-language correctness (what the caller would hear)
  * confusion matrix
  * mean detector latency

    python -m eval.run_language_detection_eval
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from voice.language_detect import LanguageDetector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="eval/datasets/language_detection_eval.json",
    )
    parser.add_argument("--out", default="results/language_detection_eval.json")
    args = parser.parse_args()

    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    detector = LanguageDetector()

    correct = 0
    confusion: dict[str, Counter] = defaultdict(Counter)
    failures = []
    latencies = []

    for row in rows:
        t0 = time.perf_counter()
        result = detector.detect(row["text"])
        latencies.append((time.perf_counter() - t0) * 1e6)

        expected = row["expected_output_language"]
        got = result.output_language
        confusion[expected][got] += 1
        if got == expected:
            correct += 1
        else:
            failures.append(
                {
                    "text": row["text"],
                    "expected": expected,
                    "got": got,
                    "detected": result.detected,
                    "reason": result.reason,
                }
            )

    n = len(rows)
    summary = {
        "n": n,
        "response_language_accuracy": round(correct / n, 4) if n else 0.0,
        "mean_detector_latency_us": round(sum(latencies) / n, 1) if n else 0.0,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "failures": failures,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Cases: {n}")
    print(f"Response-language accuracy: {summary['response_language_accuracy']:.2%}")
    print(f"Mean detector latency: {summary['mean_detector_latency_us']:.1f} us")
    print("Confusion (expected -> got):")
    for exp, got in confusion.items():
        print(f"  {exp:8s} -> {dict(got)}")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  {f}")
    print(f"Saved {args.out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
