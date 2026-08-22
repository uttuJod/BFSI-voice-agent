from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from integration.llm_router import QwenBFSIRouter

BASELINE = {
    "json_valid_rate": 1.0,
    "schema_valid_rate": 1.0,
    "intent_accuracy": 0.975,
    "tool_accuracy": 0.975,
    "rag_routing_accuracy": 1.0,
    "clarification_accuracy": 1.0,
    "argument_accuracy": 0.975,
}


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("cases", "data", "examples", "records", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Dataset must be a JSON list or a dict containing a list of cases.")


def subset_match(expected: Any, predicted: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(predicted, dict):
            return False
        return all(key in predicted and subset_match(value, predicted[key]) for key, value in expected.items())
    if isinstance(expected, list):
        return expected == predicted
    return expected == predicted


def rate(rows: list[dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows if r[key] is not None]
    return (sum(bool(v) for v in vals) / len(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--adapter", default="models/qwen35_customer_support_FINAL_FROZEN")
    parser.add_argument("--output-dir", default="results/hf_120_benchmark")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(dataset_path)
    if len(cases) != 120:
        raise SystemExit(f"Expected exactly 120 cases, found {len(cases)}")

    print("=" * 92)
    print("HF FROZEN ROUTER BENCHMARK")
    print("=" * 92)
    print("Dataset:", dataset_path)
    print("Cases:", len(cases))

    router = QwenBFSIRouter(adapter_path=args.adapter)

    rows = []
    latencies = []
    group_stats = defaultdict(list)

    print("\nRUNNING CASES")
    print("-" * 92)

    for i, case in enumerate(cases, start=1):
        case_id = case["id"]
        user = case["user"]
        group = case["group"]
        expected_intent = case["intent"]
        expected_tool = case["tool"]
        expected_rag = case["requires_rag"]
        expected_clarification = case["needs_clarification"]
        expected_args = case.get("expected_args", {})

        started = time.perf_counter()
        error = None
        try:
            decision = router.route(user)
            latency_ms = (time.perf_counter() - started) * 1000.0
            predicted = decision.model_dump()
            intent_ok = predicted["intent"] == expected_intent
            tool_ok = predicted["tool"] == expected_tool
            rag_ok = predicted["requires_rag"] == expected_rag
            clarification_ok = predicted["needs_clarification"] == expected_clarification
            arguments_ok = subset_match(expected_args, predicted.get("arguments", {}))
            json_valid = True
            schema_valid = True
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            predicted = {"intent": None, "tool": None, "requires_rag": None, "needs_clarification": None, "arguments": None}
            json_valid = schema_valid = intent_ok = tool_ok = rag_ok = clarification_ok = arguments_ok = False
            error = f"{type(exc).__name__}: {exc}"

        all_ok = all([json_valid, schema_valid, intent_ok, tool_ok, rag_ok, clarification_ok, arguments_ok])
        latencies.append(latency_ms)
        group_stats[group].append(all_ok)

        row = {
            "id": case_id,
            "group": group,
            "user": user,
            "expected_intent": expected_intent,
            "expected_tool": expected_tool,
            "expected_requires_rag": expected_rag,
            "expected_needs_clarification": expected_clarification,
            "expected_arguments": expected_args,
            "predicted_intent": predicted["intent"],
            "predicted_tool": predicted["tool"],
            "predicted_requires_rag": predicted["requires_rag"],
            "predicted_needs_clarification": predicted["needs_clarification"],
            "predicted_arguments": predicted["arguments"],
            "json_valid": json_valid,
            "schema_valid": schema_valid,
            "intent_ok": intent_ok,
            "tool_ok": tool_ok,
            "rag_ok": rag_ok,
            "clarification_ok": clarification_ok,
            "arguments_ok": arguments_ok,
            "all_ok": all_ok,
            "latency_ms": round(latency_ms, 1),
            "error": error,
        }
        rows.append(row)

        status = "PASS" if all_ok else "FAIL"
        print(f"[{i:03d}/120] {status:4s} | {case_id:22s} | {group:24s} | {latency_ms:7.1f} ms")
        if not all_ok:
            print("   USER:", user)
            print("   EXPECTED:", {"intent": expected_intent, "tool": expected_tool, "rag": expected_rag, "clarification": expected_clarification, "arguments": expected_args})
            print("   PREDICTED:", {"intent": predicted["intent"], "tool": predicted["tool"], "rag": predicted["requires_rag"], "clarification": predicted["needs_clarification"], "arguments": predicted["arguments"]})
            if error:
                print("   ERROR:", error)

    metrics = {
        "n_cases": len(rows),
        "json_valid_rate": rate(rows, "json_valid"),
        "schema_valid_rate": rate(rows, "schema_valid"),
        "intent_accuracy": rate(rows, "intent_ok"),
        "tool_accuracy": rate(rows, "tool_ok"),
        "rag_routing_accuracy": rate(rows, "rag_ok"),
        "clarification_accuracy": rate(rows, "clarification_ok"),
        "argument_accuracy": rate(rows, "arguments_ok"),
        "all_checks_pass_rate": rate(rows, "all_ok"),
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        "latency_min_ms": min(latencies),
        "latency_max_ms": max(latencies),
        "latency_mean_ms": statistics.mean(latencies),
    }

    print("\n" + "=" * 92)
    print("FINAL HF METRICS")
    print("=" * 92)
    for key in ("json_valid_rate", "schema_valid_rate", "intent_accuracy", "tool_accuracy", "rag_routing_accuracy", "clarification_accuracy", "argument_accuracy", "all_checks_pass_rate"):
        print(f"{key:28s}: {metrics[key] * 100:7.3f}%")

    print()
    print(f"{'latency_p50_ms':28s}: {metrics['latency_p50_ms']:.1f}")
    print(f"{'latency_p95_ms':28s}: {metrics['latency_p95_ms']:.1f}")
    print(f"{'latency_mean_ms':28s}: {metrics['latency_mean_ms']:.1f}")

    print("\nPER-GROUP ALL-CHECK PASS RATE")
    for group in sorted(group_stats):
        vals = group_stats[group]
        passed = sum(vals)
        print(f"{group:28s}: {passed}/{len(vals)} = {passed / len(vals) * 100:.1f}%")

    print("\nBASELINE COMPARISON")
    print("-" * 92)
    regressions = []
    for key, old in BASELINE.items():
        new = metrics[key]
        delta = new - old
        status = "OK" if new >= old else "REGRESSION"
        if status == "REGRESSION":
            regressions.append(key)
        print(f"{key:29s} old={old*100:7.3f}% new={new*100:7.3f}% delta={delta*100:+8.3f}% {status}")

    failures = [r for r in rows if not r["all_ok"]]
    (output_dir / "hf_120_summary.json").write_text(json.dumps({"dataset": str(dataset_path), "metrics": metrics, "baseline": BASELINE, "regressions": regressions}, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "hf_120_failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nREPORTS")
    print("SUMMARY :", output_dir / "hf_120_summary.json")
    print("FAILURES:", output_dir / "hf_120_failures.json")


if __name__ == "__main__":
    main()
