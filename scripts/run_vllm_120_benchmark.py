from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from integration.vllm_router import VLLMBFSIRouter, VLLMRouterError


MISSING = object()


BASELINE = {
    "n_cases": 120,
    "json_valid_rate": 1.000,
    "schema_valid_rate": 1.000,
    "intent_accuracy": 0.975,
    "tool_accuracy": 0.975,
    "rag_routing_accuracy": 1.000,
    "clarification_accuracy": 1.000,
    "argument_accuracy": 0.975,
}


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at line {lineno}: {exc}"
                    ) from exc
        return rows

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict):
            for key in ("cases", "data", "examples", "records", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value

        raise ValueError(
            "JSON dataset must be a list, or a dict containing one of: "
            "cases/data/examples/records/items."
        )

    raise ValueError("Dataset must be .json or .jsonl")


# ---------------------------------------------------------------------
# Flexible field extraction
# ---------------------------------------------------------------------

def pick(obj: dict[str, Any], *keys: str, default: Any = MISSING) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return default


def normalize_none(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {
        "none", "null", "nil", ""
    }:
        return None
    return value


def normalize_case(raw: dict[str, Any], index: int) -> dict[str, Any]:
    expected = raw.get("expected")
    if not isinstance(expected, dict):
        expected = {}

    case_id = pick(
        raw,
        "id",
        "case_id",
        "case",
        default=f"case-{index + 1:03d}",
    )

    group = pick(
        raw,
        "group",
        "family",
        "category",
        default="unknown",
    )

    user_text = pick(
        raw,
        "user",
        "text",
        "utterance",
        "input",
        "query",
        "prompt",
        default=MISSING,
    )

    if user_text is MISSING:
        messages = raw.get("messages")
        if isinstance(messages, list):
            user_messages = [
                m for m in messages
                if isinstance(m, dict) and m.get("role") == "user"
            ]
            if user_messages:
                user_text = user_messages[-1].get("content")

    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError(
            f"{case_id}: could not find user text. "
            "Expected one of user/text/utterance/input/query/prompt/messages."
        )

    expected_intent = pick(
        expected,
        "intent",
        default=pick(
            raw,
            "expected_intent",
            "intent",
            default=MISSING,
        ),
    )

    expected_tool_raw = pick(
        expected,
        "tool",
        default=pick(
            raw,
            "expected_tool",
            "tool",
            default=MISSING,
        ),
    )
    expected_tool = (
        MISSING
        if expected_tool_raw is MISSING
        else normalize_none(expected_tool_raw)
    )

    expected_rag = pick(
        expected,
        "requires_rag",
        "rag",
        default=pick(
            raw,
            "expected_requires_rag",
            "requires_rag",
            default=MISSING,
        ),
    )

    expected_clarification = pick(
        expected,
        "needs_clarification",
        "clarification",
        default=pick(
            raw,
            "expected_needs_clarification",
            "needs_clarification",
            default=MISSING,
        ),
    )

    expected_arguments = pick(
        expected,
        "arguments",
        "args",
        default=pick(
            raw,
            "expected_args",
            "expected_arguments",
            "arguments",
            default=MISSING,
        ),
    )

    return {
        "id": str(case_id),
        "group": str(group),
        "user": user_text.strip(),
        "expected_intent": expected_intent,
        "expected_tool": expected_tool,
        "expected_requires_rag": expected_rag,
        "expected_needs_clarification": expected_clarification,
        "expected_arguments": expected_arguments,
    }


# ---------------------------------------------------------------------
# Metric checks
# ---------------------------------------------------------------------

def subset_match(expected: Any, predicted: Any) -> bool:
    """
    Expected arguments are treated as required subset.

    Examples:
      expected {"amount": 4000}
      predicted {"amount": 4000, "currency": "INR"}
    => PASS

    This is safer for evaluation files where only required arguments
    are annotated.
    """
    if expected is None:
        return True

    if isinstance(expected, dict):
        if not isinstance(predicted, dict):
            return False

        for key, value in expected.items():
            if key not in predicted:
                return False
            if not subset_match(value, predicted[key]):
                return False
        return True

    if isinstance(expected, list):
        if not isinstance(predicted, list):
            return False
        return expected == predicted

    return expected == predicted


def check_optional(expected: Any, predicted: Any) -> bool | None:
    # MISSING means the dataset did not annotate this field.
    # Explicit None means "the correct expected value is null".
    if expected is MISSING:
        return None
    return expected == predicted


def rate(values: list[bool | None]) -> float | None:
    usable = [v for v in values if v is not None]
    if not usable:
        return None
    return sum(bool(v) for v in usable) / len(usable)


def fmt_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3%}"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = round(q * (len(ordered) - 1))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


# ---------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the original frozen 120-case BFSI router benchmark through vLLM."
    )
    parser.add_argument(
        "--dataset",
        default="domains/bfsi/eval/locked_120_benchmark.json",
        help=(
            "Path to the ORIGINAL locked 120-case .json or .jsonl dataset. "
            "Defaults to domains/bfsi/eval/locked_120_benchmark.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/vllm_120_benchmark",
    )
    parser.add_argument(
        "--no-structured-output",
        action="store_true",
        help="Disable vLLM schema-constrained decoding for an A/B test.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug-only case limit. Omit for the real 120-case run.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_cases = load_dataset(dataset_path)
    cases = [
        normalize_case(row, i)
        for i, row in enumerate(raw_cases)
    ]

    if args.limit is not None:
        cases = cases[: args.limit]

    print("=" * 92)
    print("VLLM FROZEN ROUTER BENCHMARK")
    print("=" * 92)
    print("Dataset:", dataset_path)
    print("Cases:", len(cases))
    print("Structured output:", not args.no_structured_output)

    if args.limit is None and len(cases) != 120:
        print()
        print(
            f"WARNING: expected the original 120-case dataset, "
            f"but loaded {len(cases)} cases."
        )
        print(
            "Do not compare this result to the published 97.5% baseline "
            "unless this is exactly the same 120-case set."
        )

    router = VLLMBFSIRouter(
        use_structured_output=not args.no_structured_output,
    )

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []

    checks = {
        "json_valid": [],
        "schema_valid": [],
        "intent": [],
        "tool": [],
        "rag": [],
        "clarification": [],
        "arguments": [],
    }

    per_group: dict[str, list[bool]] = defaultdict(list)

    try:
        print("\nHEALTH:", router.health())

        print("\nWARMUP (excluded from benchmark metrics)")
        try:
            _, warm_metrics = router.route_with_metrics("I already paid.")
            print("Warmup:", warm_metrics)
        except Exception as exc:
            print("Warmup failed:", repr(exc))

        print("\nRUNNING CASES")
        print("-" * 92)

        for idx, case in enumerate(cases, start=1):
            start = time.perf_counter()

            json_valid = False
            schema_valid = False
            error: str | None = None
            decision = None
            metrics: dict[str, Any] = {}

            try:
                decision, metrics = router.route_with_metrics(case["user"])
                json_valid = True
                schema_valid = True
            except VLLMRouterError as exc:
                error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            elapsed_ms = float(
                metrics.get(
                    "total_ms",
                    (time.perf_counter() - start) * 1000,
                )
            )
            latencies.append(elapsed_ms)

            if decision is not None:
                intent_ok = check_optional(
                    case["expected_intent"],
                    decision.intent,
                )
                tool_ok = check_optional(
                    case["expected_tool"],
                    decision.tool,
                )
                rag_ok = check_optional(
                    case["expected_requires_rag"],
                    decision.requires_rag,
                )
                clarification_ok = check_optional(
                    case["expected_needs_clarification"],
                    decision.needs_clarification,
                )

                if case["expected_arguments"] is MISSING:
                    arguments_ok = None
                else:
                    arguments_ok = subset_match(
                        case["expected_arguments"],
                        decision.arguments,
                    )

                predicted = {
                    "intent": decision.intent,
                    "tool": decision.tool,
                    "requires_rag": decision.requires_rag,
                    "requires_tool": decision.requires_tool,
                    "needs_clarification": decision.needs_clarification,
                    "arguments": decision.arguments,
                    "confidence": decision.confidence,
                    "response_style": decision.response_style,
                    "response": decision.response,
                }
            else:
                intent_ok = False if case["expected_intent"] is not None else None
                tool_ok = False if case["expected_tool"] is not None else None
                rag_ok = False if case["expected_requires_rag"] is not None else None
                clarification_ok = (
                    False
                    if case["expected_needs_clarification"] is not None
                    else None
                )
                arguments_ok = (
                    False
                    if case["expected_arguments"] is not None
                    else None
                )
                predicted = {}

            checks["json_valid"].append(json_valid)
            checks["schema_valid"].append(schema_valid)
            checks["intent"].append(intent_ok)
            checks["tool"].append(tool_ok)
            checks["rag"].append(rag_ok)
            checks["clarification"].append(clarification_ok)
            checks["arguments"].append(arguments_ok)

            required_checks = [
                json_valid,
                schema_valid,
                intent_ok,
                tool_ok,
                rag_ok,
                clarification_ok,
                arguments_ok,
            ]
            evaluated_checks = [
                x for x in required_checks
                if x is not None
            ]
            all_ok = all(evaluated_checks)

            per_group[case["group"]].append(all_ok)

            row = {
                **case,
                "predicted_intent": predicted.get("intent"),
                "predicted_tool": predicted.get("tool"),
                "predicted_requires_rag": predicted.get("requires_rag"),
                "predicted_needs_clarification": predicted.get(
                    "needs_clarification"
                ),
                "predicted_arguments": json.dumps(
                    predicted.get("arguments", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "json_valid": json_valid,
                "schema_valid": schema_valid,
                "intent_ok": intent_ok,
                "tool_ok": tool_ok,
                "rag_ok": rag_ok,
                "clarification_ok": clarification_ok,
                "arguments_ok": arguments_ok,
                "all_ok": all_ok,
                "latency_ms": round(elapsed_ms, 1),
                "prompt_tokens": metrics.get("prompt_tokens"),
                "completion_tokens": metrics.get("completion_tokens"),
                "error": error,
            }
            rows.append(row)

            mark = "PASS" if all_ok else "FAIL"

            print(
                f"[{idx:03d}/{len(cases):03d}] "
                f"{mark:<4} | "
                f"{case['id']:<20} | "
                f"{case['group']:<24} | "
                f"{elapsed_ms:7.1f} ms"
            )

            if not all_ok:
                print("   USER:", case["user"])
                print(
                    "   EXPECTED:",
                    {
                        "intent": case["expected_intent"],
                        "tool": case["expected_tool"],
                        "rag": case["expected_requires_rag"],
                        "clarification": case["expected_needs_clarification"],
                        "arguments": case["expected_arguments"],
                    },
                )
                print(
                    "   PREDICTED:",
                    {
                        "intent": predicted.get("intent"),
                        "tool": predicted.get("tool"),
                        "rag": predicted.get("requires_rag"),
                        "clarification": predicted.get(
                            "needs_clarification"
                        ),
                        "arguments": predicted.get("arguments"),
                    },
                )
                if error:
                    print("   ERROR:", error)

    finally:
        router.close()

    metric_summary = {
        "n_cases": len(cases),
        "json_valid_rate": rate(checks["json_valid"]),
        "schema_valid_rate": rate(checks["schema_valid"]),
        "intent_accuracy": rate(checks["intent"]),
        "tool_accuracy": rate(checks["tool"]),
        "rag_routing_accuracy": rate(checks["rag"]),
        "clarification_accuracy": rate(checks["clarification"]),
        "argument_accuracy": rate(checks["arguments"]),
        "all_checks_pass_rate": (
            sum(bool(r["all_ok"]) for r in rows) / len(rows)
            if rows
            else None
        ),
        "latency_p50_ms": statistics.median(latencies)
        if latencies else None,
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_min_ms": min(latencies) if latencies else None,
        "latency_max_ms": max(latencies) if latencies else None,
        "latency_mean_ms": statistics.mean(latencies)
        if latencies else None,
    }

    print("\n" + "=" * 92)
    print("FINAL VLLM METRICS")
    print("=" * 92)

    for key in (
        "json_valid_rate",
        "schema_valid_rate",
        "intent_accuracy",
        "tool_accuracy",
        "rag_routing_accuracy",
        "clarification_accuracy",
        "argument_accuracy",
        "all_checks_pass_rate",
    ):
        print(f"{key:<28}: {fmt_rate(metric_summary[key])}")

    print()
    print(
        f"{'latency_p50_ms':<28}: "
        f"{metric_summary['latency_p50_ms']:.1f}"
    )
    print(
        f"{'latency_p95_ms':<28}: "
        f"{metric_summary['latency_p95_ms']:.1f}"
    )
    print(
        f"{'latency_mean_ms':<28}: "
        f"{metric_summary['latency_mean_ms']:.1f}"
    )

    print("\nPER-GROUP ALL-CHECK PASS RATE")
    for group in sorted(per_group):
        vals = per_group[group]
        print(
            f"{group:<28}: "
            f"{sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.1%}"
        )

    print("\nBASELINE COMPARISON")
    print("-" * 92)
    baseline_map = {
        "json_valid_rate": BASELINE["json_valid_rate"],
        "schema_valid_rate": BASELINE["schema_valid_rate"],
        "intent_accuracy": BASELINE["intent_accuracy"],
        "tool_accuracy": BASELINE["tool_accuracy"],
        "rag_routing_accuracy": BASELINE["rag_routing_accuracy"],
        "clarification_accuracy": BASELINE["clarification_accuracy"],
        "argument_accuracy": BASELINE["argument_accuracy"],
    }

    regressions = []

    for key, old in baseline_map.items():
        new = metric_summary[key]

        if new is None:
            status = "N/A"
            delta = None
        else:
            delta = new - old
            status = "OK" if new >= old else "REGRESSION"
            if new < old:
                regressions.append(key)

        delta_text = (
            "N/A"
            if delta is None
            else f"{delta:+.3%}"
        )

        print(
            f"{key:<28} "
            f"old={old:.3%} "
            f"new={fmt_rate(new):>9} "
            f"delta={delta_text:>9} "
            f"{status}"
        )

    # -----------------------------------------------------------------
    # Save reports
    # -----------------------------------------------------------------

    csv_path = output_dir / "vllm_120_predictions.csv"
    json_path = output_dir / "vllm_120_summary.json"
    failures_path = output_dir / "vllm_120_failures.json"

    if rows:
        with csv_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    summary_payload = {
        "dataset": str(dataset_path),
        "structured_output": not args.no_structured_output,
        "metrics": metric_summary,
        "baseline": BASELINE,
        "regressions": regressions,
        "safe_to_consider_switching": (
            len(cases) == 120
            and not regressions
        ),
    }

    json_path.write_text(
        json.dumps(
            summary_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    failures = [
        row
        for row in rows
        if not row["all_ok"]
    ]

    failures_path.write_text(
        json.dumps(
            failures,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nREPORTS")
    print("CSV     :", csv_path)
    print("SUMMARY :", json_path)
    print("FAILURES:", failures_path)

    print("\nDECISION")
    print("-" * 92)

    if len(cases) != 120:
        print(
            "NOT VALID FOR PRODUCTION COMPARISON: "
            "this was not exactly 120 cases."
        )
        sys.exit(2)

    if regressions:
        print(
            "DO NOT SWITCH THE PRODUCTION ROUTER YET."
        )
        print(
            "Regressed metrics:",
            ", ".join(regressions),
        )
        sys.exit(3)

    print(
        "NO BENCHMARK METRIC REGRESSION DETECTED."
    )
    print(
        "This backend can move to the next integration test, "
        "but still keep the HF router available for rollback."
    )


if __name__ == "__main__":
    main()
