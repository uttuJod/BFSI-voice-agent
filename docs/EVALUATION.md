# Evaluation and Regression Coverage

This project uses layered regression testing rather than relying only on one end-to-end demo. Every number in this document comes from a command listed next to it.

## Which numbers are blind

Honesty about evaluation is part of the evaluation. The categories:

| Dataset | Status | Meaning |
|---|---|---|
| `domains/bfsi/eval/locked_120_benchmark.json` (router) | locked before the final fine-tune stage | Model was not trained on it; prompts and decoding were frozen against it |
| `domains/bfsi/eval/rag_eval.json` (110 cases) | development | RAG thresholds were tuned on it; treat as validation accuracy |
| `domains/bfsi/eval/rag_eval_holdout.json` (100 cases) | tuned against | First untouched run scored 56%; failures were inspected and v4/v4.1 fixes were made. The 94% figure is therefore a validation number, not blind |
| `domains/ecommerce/eval/rag_eval.json` (40 cases) | never tuned against | Written after the RAG was frozen, on the evaluators' corpus. Run once with `DOMAIN=ecommerce make eval-rag` and record the result as the blind number |
| `eval/datasets/language_detection_eval.json` (54) | lexicon tuned on it | Report as development accuracy; the four evaluator utterances are included verbatim |

Rule from here on: `make eval-rag-blind` and the e-commerce eval are run once per release and the result is pasted below. They are not used to change thresholds.

## How hallucination is scored

A response counts as a hallucination when either holds:

1. It contains a `forbidden_claims` string for that case (e.g. "10 days" on a question whose active policy says 7 days, "Paris" for an out-of-scope question, "cancelled" for an order the corpus knows nothing about).
2. It answers (`answerable=True`, verdict SUFFICIENT) a case whose expected behaviour is abstain.

Fact coverage is scored separately: an answer passes only if every `expected_facts` string appears. Source attribution passes only if cited document ids include every `expected_sources` id.

This is a rule-based judge, which is deterministic and reproducible but conservative: it catches wrong numbers and fabricated outcomes, not subtle paraphrase errors. Cases are written so that the decisive fact is a number, a named condition or an abstention, which the rule judge can check exactly. An LLM judge was deliberately not used so the reported rate cannot drift with the judge model.

## Baseline vs self-correcting

`make eval-rag` runs both systems on the same cases and prints them side by side with accuracy, hallucination rate, retrieval recall, source attribution, clarification, abstention, conflict detection, conflict resolution, latency P50/P95 and mean retrieval iterations. Output files: `results/rag_eval_summary_<name>.json`, `results/rag_eval_detailed_<name>.json`, `results/rag_failures_<name>.json`.

## Language detection

`make eval-lid` (development accuracy: 54/54, mean 4 µs per detection). Confusion matrix and failures are written to `results/language_detection_eval.json`.

## Voice latency and barge-in

Every voice turn appends a record to `results/voice_turn_samples.jsonl` with stage timings: `router_ms`, `guards_ms`, `rag_ms`, `tool_ms`, `response_builder_ms`, `orchestrator_ms`, `localization_ms`, `tts_first_audio_ms`, `speech_end_to_first_audio_ms`. Every confirmed barge-in appends a stop-latency sample and every rejected candidate appends a reason to `results/barge_in_samples.jsonl`.

`make bench` prints the per-stage P50/P95 table against the Problem 4 targets, end-to-end latency by output language, and the barge-in table (P50, P95, share within 500 ms, suppressed false candidates by reason). Record at least 50 turns per language and 20 barge-ins before quoting the table.

## Guaranteed delivery

`make test` covers the pipeline in-process (9 tests). `make chaos` runs four scripts against real separate worker processes: duplicate submission, SIGKILL mid-job with reclaim, store restart with queued jobs, permanent failure to DLQ with backoff. Throughput from the unit test is written to `results/jobs_throughput.json`.

## Router benchmark

Latest locked 120-case evaluation snapshot:

| Metric | Result |
|---|---:|
| Cases | 120 |
| JSON validity | 100% |
| Schema validity | 100% |
| Intent accuracy | 98.33% |
| Tool accuracy | 98.33% |
| RAG routing accuracy | 100% |
| Clarification accuracy | 100% |
| Argument accuracy | 98.33% |
| All-check accuracy | 98.33% |

The two remaining router misses are own-account identity-verification phrasings. Production orchestration applies a deterministic normalization guard for those cases rather than changing the frozen router prompt or decoding settings.

## Identity verification regressions

### `scripts.test_identity_session_flow`

Covers:

- unverified protected-account access is blocked
- no account data is returned before verification
- verification challenge creation
- wrong-code attempt handling
- correct-code acceptance
- session verification state
- automatic resumption of the original request
- protected account access succeeds only after verification

Expected final line:

```text
IDENTITY SESSION FLOW: PASS
```

### `scripts.test_verification_speech_parser`

Covers deterministic spoken-digit parsing:

```text
0641
0 6 4 1
Zero six four one
zero, six, four, one.
oh six four one
```

Expected final line:

```text
VERIFICATION SPEECH PARSER: PASS
```

### `scripts.test_verification_currency_artifact`

Covers recovery from an observed streaming-STT smart-format artifact where a spoken digit sequence may be rendered as a currency-like value during verification.

Examples covered:

```text
$64.01 -> 0641
64.01  -> 0641
```

This normalization is intentionally narrow and verification-specific; it does not globally reinterpret arbitrary money values.

## RAG regression

### `scripts.test_rag_active_version`

Checks that the current policy is used for normal present-tense policy questions and that active-version citations are returned.

Expected final line:

```text
RAG ACTIVE-VERSION FILTER: PASS
```

## Voice end-to-end checks

Recommended manual smoke sequence:

1. Ask for current outstanding balance.
2. Confirm that the agent requests identity verification before disclosing account data.
3. Speak the verification digits naturally rather than rushing.
4. Confirm successful verification.
5. Confirm the original balance request resumes automatically.
6. Ask a policy question such as the grace-period rule.
7. Disconnect/reconnect and confirm verification is required again for protected data.

## Regression philosophy

- Do not change frozen router inference settings merely to address a downstream integration issue.
- Keep model-routing regressions separate from RAG/tool/voice regressions.
- Prefer deterministic guards for narrow policy/privacy/business invariants.
- Re-run the relevant benchmark after touching routing/orchestration behavior.
- Re-run RAG regressions after changing retrieval, version filtering, evidence evaluation, or generation evidence.
- Re-run identity tests after changing session handling, protected-tool gating, or verification parsing.
