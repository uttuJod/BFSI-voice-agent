"""
Aggregate per-turn voice latency samples and barge-in samples into the
per-stage P50/P95 table used in docs/EVALUATION.md.

Inputs (written automatically by the voice runtime):
  results/voice_turn_samples.jsonl
  results/barge_in_samples.jsonl

    python -m bench.latency_report
    python -m bench.latency_report --markdown > results/latency_table.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TARGETS_MS = {
    "router_ms": 2000,                 # PS4: LLM first token < 2 s
    "tts_first_audio_ms": 1000,        # PS4: TTS first audio < 1 s
    "speech_end_to_first_audio_ms": 4000,  # PS4: end-to-end < 4 s
}

STAGE_ORDER = [
    "router_ms",
    "guards_ms",
    "rag_ms",
    "tool_ms",
    "response_builder_ms",
    "orchestrator_ms",
    "localization_ms",
    "tts_first_audio_ms",
    "speech_end_to_first_audio_ms",
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def stage_table(turns: list[dict]) -> list[dict]:
    values: dict[str, list[float]] = defaultdict(list)
    for t in turns:
        for k, v in (t.get("stages") or {}).items():
            if isinstance(v, (int, float)):
                values[k].append(float(v))
    rows = []
    for stage in STAGE_ORDER + sorted(set(values) - set(STAGE_ORDER)):
        vals = values.get(stage)
        if not vals:
            continue
        target = TARGETS_MS.get(stage)
        p95 = percentile(vals, 0.95)
        rows.append(
            {
                "stage": stage,
                "n": len(vals),
                "p50_ms": round(percentile(vals, 0.5), 1),
                "p95_ms": round(p95, 1),
                "max_ms": round(max(vals), 1),
                "target_ms": target,
                "meets_target_p95": (p95 <= target) if target else None,
            }
        )
    return rows


def by_language(turns: list[dict]) -> dict[str, dict]:
    out: dict[str, list[float]] = defaultdict(list)
    for t in turns:
        v = (t.get("stages") or {}).get("speech_end_to_first_audio_ms")
        if isinstance(v, (int, float)):
            out[t.get("output_language", "unknown")].append(float(v))
    return {
        lang: {
            "n": len(v),
            "p50_ms": round(percentile(v, 0.5), 1),
            "p95_ms": round(percentile(v, 0.95), 1),
        }
        for lang, v in out.items()
    }


def barge_summary(samples: list[dict]) -> dict:
    confirmed = [s["stop_latency_ms"] for s in samples if "stop_latency_ms" in s]
    rejected = [s for s in samples if s.get("rejected")]
    return {
        "confirmed_barge_ins": len(confirmed),
        "p50_ms": round(percentile(confirmed, 0.5), 1),
        "p95_ms": round(percentile(confirmed, 0.95), 1),
        "max_ms": round(max(confirmed), 1) if confirmed else 0.0,
        "target_ms": 500,
        "within_target_rate": (
            round(sum(v <= 500 for v in confirmed) / len(confirmed), 3)
            if confirmed
            else None
        ),
        "false_candidates_suppressed": len(rejected),
        "suppression_reasons": dict(
            sorted(
                defaultdict(
                    int,
                    {r: sum(1 for s in rejected if s.get("reason") == r)
                     for r in {s.get("reason") for s in rejected}},
                ).items()
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", default="results/voice_turn_samples.jsonl")
    parser.add_argument("--barge", default="results/barge_in_samples.jsonl")
    parser.add_argument("--out", default="results/latency_report.json")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    turns = load_jsonl(Path(args.turns))
    barge = load_jsonl(Path(args.barge))

    report = {
        "turns": len(turns),
        "stages": stage_table(turns),
        "e2e_by_output_language": by_language(turns),
        "barge_in": barge_summary(barge),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    if args.markdown:
        print(f"Turns measured: {len(turns)}\n")
        print("| Stage | n | P50 ms | P95 ms | Max ms | Target ms | P95 meets target |")
        print("|---|---:|---:|---:|---:|---:|:---:|")
        for r in report["stages"]:
            tgt = r["target_ms"] if r["target_ms"] else ""
            ok = "" if r["meets_target_p95"] is None else ("yes" if r["meets_target_p95"] else "no")
            print(f"| {r['stage']} | {r['n']} | {r['p50_ms']} | {r['p95_ms']} | {r['max_ms']} | {tgt} | {ok} |")
        print("\n| Output language | n | E2E P50 ms | E2E P95 ms |")
        print("|---|---:|---:|---:|")
        for lang, v in report["e2e_by_output_language"].items():
            print(f"| {lang} | {v['n']} | {v['p50_ms']} | {v['p95_ms']} |")
        b = report["barge_in"]
        print("\n| Barge-in metric | Value |")
        print("|---|---:|")
        print(f"| Confirmed barge-ins | {b['confirmed_barge_ins']} |")
        print(f"| Stop latency P50 ms | {b['p50_ms']} |")
        print(f"| Stop latency P95 ms | {b['p95_ms']} |")
        print(f"| Within 500 ms target | {b['within_target_rate']} |")
        print(f"| False candidates suppressed | {b['false_candidates_suppressed']} |")
    else:
        print(json.dumps(report, indent=2))

    if not turns:
        print("\nNo voice turns recorded yet. Run a voice session first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
