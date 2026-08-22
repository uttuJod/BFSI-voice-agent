from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from eval.schemas import EvalRecord
from rag import BaselineRAG, RAGConfig, SelfCorrectingRAG
from rag.config import domain_path, active_domain


RESULTS_DIR = Path("results")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate baseline and self-correcting RAG."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Path to evaluation JSON dataset. Defaults to "
            "domains/<DOMAIN>/eval/rag_eval.json."
        ),
    )

    parser.add_argument(
        "--name",
        type=str,
        default="development",
        help="Name used for output result files.",
    )

    return parser.parse_args()


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_mean(values: list[float]) -> float:

    if not values:
        return 0.0

    return float(
        mean(values)
    )


def safe_rate(values: list[bool]) -> float:

    if not values:
        return 0.0

    return (
        sum(bool(value) for value in values)
        / len(values)
    )


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:

    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return float(values[0])

    position = (
        (len(values) - 1)
        * percentile_value
    )

    lower_index = int(position)

    upper_index = min(
        lower_index + 1,
        len(values) - 1,
    )

    fraction = (
        position
        - lower_index
    )

    lower = values[lower_index]
    upper = values[upper_index]

    return float(
        lower
        + (
            (upper - lower)
            * fraction
        )
    )


# ============================================================
# SOURCE HELPERS
# ============================================================

def actual_source_ids(result) -> set[str]:

    return {
        citation.document_id
        for citation in result.citations
    }


def retrieved_source_ids(result) -> set[str]:

    return {
        chunk.document_id
        for chunk in result.retrieved_chunks
    }


def expected_source_ids(
    record: EvalRecord,
) -> set[str]:

    return set(
        record.expected_sources
    )


def normalized_answer(result) -> str:

    return (
        result.answer
        or ""
    ).strip().lower()


# ============================================================
# BEHAVIOR
# ============================================================

def behavior_correct(
    record: EvalRecord,
    result,
) -> bool:

    expected = (
        record.expected_behavior
    )

    if expected == "answer":

        return (
            result.answerable
            and bool(result.answer)
            and result.verdict.value
            == "SUFFICIENT"
        )

    if expected == "abstain":

        return (
            not result.answerable
            and result.verdict.value
            in {
                "INSUFFICIENT",
                "OUT_OF_SCOPE",
                "CONFLICT",
            }
        )

    if expected == "clarify":

        return (
            not result.answerable
            and result.verdict.value
            == "AMBIGUOUS"
            and bool(
                result.clarification_question
            )
        )

    if expected == "conflict":

        return bool(
            result.conflict_detected
        )

    raise ValueError(
        f"Unknown expected_behavior: {expected}"
    )


# ============================================================
# FACT ACCURACY
# ============================================================

def expected_fact_coverage(
    record: EvalRecord,
    result,
) -> float:

    if not record.expected_facts:
        return 1.0

    answer = normalized_answer(
        result
    )

    if not answer:
        return 0.0

    hits = 0

    for expected_fact in (
        record.expected_facts
    ):

        if (
            expected_fact.lower()
            in answer
        ):
            hits += 1

    return (
        hits
        / len(record.expected_facts)
    )


def answer_fact_correct(
    record: EvalRecord,
    result,
) -> bool:

    if (
        record.expected_behavior
        != "answer"
    ):
        return True

    if not record.expected_facts:
        return bool(
            result.answer
        )

    return (
        expected_fact_coverage(
            record,
            result,
        )
        > 0.0
    )


# ============================================================
# SOURCE ATTRIBUTION
# ============================================================

def source_attribution_correct(
    record: EvalRecord,
    result,
) -> bool:

    expected = expected_source_ids(
        record
    )

    if not expected:
        return True

    if not result.answerable:
        return False

    actual = actual_source_ids(
        result
    )

    return bool(
        actual.intersection(
            expected
        )
    )


# ============================================================
# RETRIEVAL RECALL
# ============================================================

def retrieval_recall_at_k(
    record: EvalRecord,
    result,
) -> float:

    expected = expected_source_ids(
        record
    )

    if not expected:
        return 1.0

    retrieved = retrieved_source_ids(
        result
    )

    return (
        len(
            expected.intersection(
                retrieved
            )
        )
        / len(expected)
    )


# ============================================================
# HALLUCINATION
# ============================================================

def forbidden_claim_violation(
    record: EvalRecord,
    result,
) -> bool:

    answer = normalized_answer(
        result
    )

    if not answer:
        return False

    return any(
        claim.lower()
        in answer

        for claim
        in record.forbidden_claims
    )


# ============================================================
# ABSTENTION
# ============================================================

def correct_abstention(
    record: EvalRecord,
    result,
) -> bool | None:

    if (
        record.expected_behavior
        != "abstain"
    ):
        return None

    return (
        not result.answerable
        and result.verdict.value
        in {
            "INSUFFICIENT",
            "OUT_OF_SCOPE",
            "CONFLICT",
        }
    )


# ============================================================
# CLARIFICATION
# ============================================================

def clarification_correct(
    record: EvalRecord,
    result,
) -> bool | None:

    if (
        record.expected_behavior
        != "clarify"
    ):
        return None

    return (
        result.verdict.value
        == "AMBIGUOUS"

        and not result.answerable

        and bool(
            result.clarification_question
        )
    )


# ============================================================
# CONFLICTS
# ============================================================

CONFLICT_CATEGORIES = {
    "conflicting_policies",
    "outdated_document_conflict",
    "outdated_document_conflicts",
    "conflict",
}


def is_conflict_case(
    record: EvalRecord,
) -> bool:

    return (
        record.category
        in CONFLICT_CATEGORIES

        or record.expected_behavior
        == "conflict"
    )


def conflict_detection_correct(
    record: EvalRecord,
    result,
) -> bool | None:

    if not is_conflict_case(
        record
    ):
        return None

    return bool(
        result.conflict_detected
    )


def conflict_resolution_correct(
    record: EvalRecord,
    result,
) -> bool | None:

    if not is_conflict_case(
        record
    ):
        return None

    conflict = (
        result.conflict_resolution
    )

    if conflict is None:
        return False

    expected = expected_source_ids(
        record
    )

    if record.answerable:

        if (
            conflict.resolution_status.value
            != "resolved"
        ):
            return False

        if not expected:
            return True

        return (
            conflict.preferred_document_id
            in expected
        )

    return (
        conflict.resolution_status.value
        == "unresolved"

        and not result.answerable
    )


# ============================================================
# FAILURE CLASSIFICATION
# ============================================================

def classify_failures(
    record: EvalRecord,
    result,
    behavior_ok: bool,
    source_ok: bool,
    fact_ok: bool,
    hallucination: bool,
    recall: float,
    conflict_detection_ok: bool | None,
    conflict_resolution_ok: bool | None,
) -> list[str]:

    failures = []

    if (
        record.expected_sources
        and recall == 0.0
    ):
        failures.append(
            "retrieval_miss"
        )

    if (
        record.expected_sources
        and result.retrieved_chunks
        and recall == 0.0
    ):
        failures.append(
            "irrelevant_retrieval"
        )

    if hallucination:
        failures.append(
            "unsupported_answer"
        )

    if (
        conflict_detection_ok
        is False
    ):
        failures.append(
            "missed_conflict"
        )

    if (
        conflict_resolution_ok
        is False
    ):
        failures.append(
            "wrong_conflict_resolution"
        )

    if (
        record.expected_behavior
        == "answer"

        and not result.answerable
    ):
        failures.append(
            "unnecessary_abstention"
        )

    if (
        record.expected_behavior
        == "clarify"

        and result.verdict.value
        != "AMBIGUOUS"
    ):
        failures.append(
            "should_have_clarified"
        )

    if (
        record.expected_sources
        and not source_ok
    ):
        failures.append(
            "wrong_source"
        )

    if not fact_ok:
        failures.append(
            "answer_fact_miss"
        )

    if (
        not behavior_ok
        and not failures
    ):
        failures.append(
            "behavior_mismatch"
        )

    return failures


# ============================================================
# EVALUATE SYSTEM
# ============================================================

async def evaluate_system(
    system_name: str,
    rag,
    records: list[EvalRecord],
):

    rows = []
    failures = []

    for record in records:

        result = await rag.answer(
            record.question
        )

        behavior_ok = (
            behavior_correct(
                record,
                result,
            )
        )

        source_ok = (
            source_attribution_correct(
                record,
                result,
            )
        )

        fact_coverage = (
            expected_fact_coverage(
                record,
                result,
            )
        )

        fact_ok = (
            answer_fact_correct(
                record,
                result,
            )
        )

        hallucination = (
            forbidden_claim_violation(
                record,
                result,
            )
        )

        recall = (
            retrieval_recall_at_k(
                record,
                result,
            )
        )

        abstention_ok = (
            correct_abstention(
                record,
                result,
            )
        )

        clarification_ok = (
            clarification_correct(
                record,
                result,
            )
        )

        conflict_detection_ok = (
            conflict_detection_correct(
                record,
                result,
            )
        )

        conflict_resolution_ok = (
            conflict_resolution_correct(
                record,
                result,
            )
        )

        passed = (
            behavior_ok
            and source_ok
            and fact_ok
            and not hallucination
        )

        rows.append(
            {
                "id": record.id,
                "system": system_name,
                "category": record.category,
                "question": record.question,

                "passed": passed,

                "expected_behavior":
                    record.expected_behavior,

                "actual_verdict":
                    result.verdict.value,

                "answerable":
                    result.answerable,

                "behavior_correct":
                    behavior_ok,

                "source_attribution_correct":
                    source_ok,

                "expected_fact_coverage":
                    fact_coverage,

                "answer_fact_correct":
                    fact_ok,

                "hallucination":
                    hallucination,

                "retrieval_recall_at_k":
                    recall,

                "correct_abstention":
                    abstention_ok,

                "clarification_correct":
                    clarification_ok,

                "conflict_detection_correct":
                    conflict_detection_ok,

                "conflict_resolution_correct":
                    conflict_resolution_ok,

                "retrieval_iterations":
                    result.retrieval_iterations,

                "latency_ms":
                    result.latency.total_ms,

                "retrieved_documents": [
                    chunk.document_id

                    for chunk
                    in result.retrieved_chunks
                ],

                "citations": [
                    citation.document_id

                    for citation
                    in result.citations
                ],

                "answer":
                    result.answer,

                "correction_actions": [
                    action.value

                    for action
                    in result.correction_actions
                ],
            }
        )

        if not passed:

            reasons = classify_failures(
                record,
                result,
                behavior_ok,
                source_ok,
                fact_ok,
                hallucination,
                recall,
                conflict_detection_ok,
                conflict_resolution_ok,
            )

            failures.append(
                {
                    "system":
                        system_name,

                    "id":
                        record.id,

                    "category":
                        record.category,

                    "question":
                        record.question,

                    "expected_behavior":
                        record.expected_behavior,

                    "actual_behavior":
                        result.verdict.value,

                    "answer":
                        result.answer,

                    "retrieved_docs": [
                        chunk.document_id

                        for chunk
                        in result.retrieved_chunks
                    ],

                    "citations": [
                        citation.document_id

                        for citation
                        in result.citations
                    ],

                    "reason":
                        reasons,

                    "correction_history": [
                        action.value

                        for action
                        in result.correction_actions
                    ],
                }
            )

    return (
        rows,
        failures,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    rows: list[dict[str, Any]],
):

    abstention_values = [
        row["correct_abstention"]

        for row in rows

        if row[
            "correct_abstention"
        ] is not None
    ]

    clarification_values = [
        row[
            "clarification_correct"
        ]

        for row in rows

        if row[
            "clarification_correct"
        ] is not None
    ]

    conflict_detection_values = [
        row[
            "conflict_detection_correct"
        ]

        for row in rows

        if row[
            "conflict_detection_correct"
        ] is not None
    ]

    conflict_resolution_values = [
        row[
            "conflict_resolution_correct"
        ]

        for row in rows

        if row[
            "conflict_resolution_correct"
        ] is not None
    ]

    latencies = [
        row["latency_ms"]
        for row in rows
    ]

    return {
        "answer_accuracy":
            safe_rate(
                [
                    row["passed"]
                    for row in rows
                ]
            ),

        "hallucination_rate":
            safe_rate(
                [
                    row["hallucination"]
                    for row in rows
                ]
            ),

        "correct_abstention_rate":
            safe_rate(
                abstention_values
            ),

        "clarification_accuracy":
            safe_rate(
                clarification_values
            ),

        "conflict_detection_accuracy":
            safe_rate(
                conflict_detection_values
            ),

        "conflict_resolution_accuracy":
            safe_rate(
                conflict_resolution_values
            ),

        "retrieval_recall_at_k":
            safe_mean(
                [
                    row[
                        "retrieval_recall_at_k"
                    ]

                    for row in rows
                ]
            ),

        "source_attribution_accuracy":
            safe_rate(
                [
                    row[
                        "source_attribution_correct"
                    ]

                    for row in rows
                ]
            ),

        "avg_retrieval_iterations":
            safe_mean(
                [
                    row[
                        "retrieval_iterations"
                    ]

                    for row in rows
                ]
            ),

        "p50_latency_ms":
            percentile(
                latencies,
                0.50,
            ),

        "p95_latency_ms":
            percentile(
                latencies,
                0.95,
            ),
    }


# ============================================================
# CATEGORY REPORT
# ============================================================

def category_report(
    rows,
):

    grouped = defaultdict(
        list
    )

    for row in rows:

        grouped[
            row["category"]
        ].append(
            row
        )

    report = {}

    for category, items in (
        sorted(grouped.items())
    ):

        report[category] = {
            "n_cases":
                len(items),

            "accuracy":
                safe_rate(
                    [
                        row["passed"]
                        for row in items
                    ]
                ),

            "retrieval_recall_at_k":
                safe_mean(
                    [
                        row[
                            "retrieval_recall_at_k"
                        ]

                        for row in items
                    ]
                ),

            "avg_retrieval_iterations":
                safe_mean(
                    [
                        row[
                            "retrieval_iterations"
                        ]

                        for row in items
                    ]
                ),
        }

    return report


# ============================================================
# DISPLAY
# ============================================================

PERCENT_METRICS = {
    "answer_accuracy",
    "hallucination_rate",
    "correct_abstention_rate",
    "clarification_accuracy",
    "conflict_detection_accuracy",
    "conflict_resolution_accuracy",
    "retrieval_recall_at_k",
    "source_attribution_accuracy",
}


DISPLAY_NAMES = {
    "answer_accuracy":
        "Answer accuracy",

    "hallucination_rate":
        "Hallucination rate",

    "correct_abstention_rate":
        "Correct abstention",

    "clarification_accuracy":
        "Clarification accuracy",

    "conflict_detection_accuracy":
        "Conflict detection",

    "conflict_resolution_accuracy":
        "Conflict resolution",

    "retrieval_recall_at_k":
        "Retrieval recall@k",

    "source_attribution_accuracy":
        "Source attribution",

    "avg_retrieval_iterations":
        "Avg retrieval passes",

    "p50_latency_ms":
        "P50 latency",

    "p95_latency_ms":
        "P95 latency",
}


def print_metrics(
    baseline,
    enhanced,
):

    print()
    print("=" * 80)
    print("RAG EVALUATION")
    print("=" * 80)

    print(
        f"{'Metric':32}"
        f"{'Baseline':>22}"
        f"{'Self-Correcting':>24}"
    )

    print("-" * 80)

    for key in baseline:

        label = DISPLAY_NAMES.get(
            key,
            key,
        )

        b = baseline[key]
        s = enhanced[key]

        if key in PERCENT_METRICS:

            b_text = (
                f"{b * 100:.1f}%"
            )

            s_text = (
                f"{s * 100:.1f}%"
            )

        elif "latency" in key:

            b_text = (
                f"{b:.2f} ms"
            )

            s_text = (
                f"{s:.2f} ms"
            )

        else:

            b_text = (
                f"{b:.2f}"
            )

            s_text = (
                f"{s:.2f}"
            )

        print(
            f"{label:32}"
            f"{b_text:>22}"
            f"{s_text:>24}"
        )

    print("=" * 80)


def print_failure_summary(
    failures,
):

    print()
    print(
        "FAILURE ANALYSIS"
    )

    print("-" * 80)

    for system in [
        "baseline",
        "self_correcting",
    ]:

        system_failures = [
            failure

            for failure in failures

            if failure["system"]
            == system
        ]

        counts = Counter()

        for failure in (
            system_failures
        ):

            counts.update(
                failure["reason"]
            )

        print()

        print(
            f"{system}: "
            f"{len(system_failures)} "
            f"failed cases"
        )

        if not counts:

            print(
                "  No failures."
            )

            continue

        for failure_type, count in (
            counts.most_common()
        ):

            print(
                f"  {failure_type}: "
                f"{count}"
            )


# ============================================================
# MAIN
# ============================================================

async def main():

    args = parse_args()

    dataset_path = (
        Path(args.dataset)
        if args.dataset
        else domain_path("eval", "rag_eval.json")
    )

    print(f"Domain: {active_domain()}")

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{dataset_path}"
        )

    raw = json.loads(
        dataset_path.read_text(
            encoding="utf-8"
        )
    )

    records = [
        EvalRecord.model_validate(
            item
        )

        for item in raw
    ]

    print(
        f"Loaded {len(records)} "
        f"evaluation cases."
    )

    print(
        f"Dataset: {dataset_path}"
    )

    config = RAGConfig()

    baseline = BaselineRAG(
        config
    )

    enhanced = SelfCorrectingRAG(
        config
    )

    baseline_rows, baseline_failures = (
        await evaluate_system(
            "baseline",
            baseline,
            records,
        )
    )

    enhanced_rows, enhanced_failures = (
        await evaluate_system(
            "self_correcting",
            enhanced,
            records,
        )
    )

    baseline_metrics = (
        calculate_metrics(
            baseline_rows
        )
    )

    enhanced_metrics = (
        calculate_metrics(
            enhanced_rows
        )
    )

    print_metrics(
        baseline_metrics,
        enhanced_metrics,
    )

    failures = (
        baseline_failures
        + enhanced_failures
    )

    print_failure_summary(
        failures
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_name = re_safe_name(
        args.name
    )

    summary_path = (
        RESULTS_DIR
        / f"rag_eval_summary_{safe_name}.json"
    )

    failures_path = (
        RESULTS_DIR
        / f"rag_failures_{safe_name}.json"
    )

    detailed_path = (
        RESULTS_DIR
        / f"rag_eval_detailed_{safe_name}.json"
    )

    summary = {
        "dataset":
            str(dataset_path),

        "name":
            args.name,

        "n_cases":
            len(records),

        "baseline":
            baseline_metrics,

        "self_correcting":
            enhanced_metrics,

        "category_metrics": {
            "baseline":
                category_report(
                    baseline_rows
                ),

            "self_correcting":
                category_report(
                    enhanced_rows
                ),
        },
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    failures_path.write_text(
        json.dumps(
            failures,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    detailed_path.write_text(
        json.dumps(
            {
                "baseline":
                    baseline_rows,

                "self_correcting":
                    enhanced_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()

    print(
        f"Saved {failures_path}"
    )

    print(
        f"Saved {summary_path}"
    )

    print(
        f"Saved {detailed_path}"
    )


def re_safe_name(
    value: str,
) -> str:

    import re

    value = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        value.strip(),
    )

    return (
        value
        or "evaluation"
    )


if __name__ == "__main__":
    asyncio.run(
        main()
    )