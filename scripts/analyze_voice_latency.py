from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean


PERF_RE = re.compile(
    r"PERF\s*\|\s*"
    r"orchestrator_ms=(?P<orchestrator>\d+(?:\.\d+)?)\s*\|\s*"
    r"localization_ms=(?P<localization>\d+(?:\.\d+)?)\s*\|\s*"
    r"tts_first_audio_ms=(?P<tts>\d+(?:\.\d+)?)\s*\|\s*"
    r"speech_end_to_first_audio_ms=(?P<e2e>\d+(?:\.\d+)?)",
    flags=re.MULTILINE,
)


def read_log(path: Path) -> str:
    raw = path.read_bytes()

    # PowerShell Tee-Object commonly writes UTF-16LE depending on host/version.
    if raw.startswith(b"\xff\xfe") or b"\x00" in raw[:200]:
        try:
            return raw.decode("utf-16")
        except UnicodeError:
            return raw.decode("utf-16-le", errors="replace")

    try:
        return raw.decode("utf-8-sig")
    except UnicodeError:
        return raw.decode("utf-8", errors="replace")


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)

    if low == high:
        return ordered[low]

    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "min_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
            "mean_ms": None,
        }

    return {
        "count": len(values),
        "min_ms": round(min(values), 1),
        "p50_ms": round(percentile(values, 0.50), 1),
        "p95_ms": round(percentile(values, 0.95), 1),
        "max_ms": round(max(values), 1),
        "mean_ms": round(mean(values), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("results/voice_latency_summary.json"),
    )
    args = parser.parse_args()

    text = read_log(args.log_file)

    rows = [
        {
            "orchestrator_ms": float(m.group("orchestrator")),
            "localization_ms": float(m.group("localization")),
            "tts_first_audio_ms": float(m.group("tts")),
            "speech_end_to_first_audio_ms": float(m.group("e2e")),
        }
        for m in PERF_RE.finditer(text)
    ]

    summary = {
        "n": len(rows),
        "orchestrator": summarize([r["orchestrator_ms"] for r in rows]),
        "localization": summarize([r["localization_ms"] for r in rows]),
        "tts_first_audio": summarize([r["tts_first_audio_ms"] for r in rows]),
        "speech_end_to_first_audio": summarize(
            [r["speech_end_to_first_audio_ms"] for r in rows]
        ),
        "samples": rows,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {args.json_out}")


if __name__ == "__main__":
    main()
