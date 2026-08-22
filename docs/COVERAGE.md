# Coverage Map: Evaluation Problem Statements

How each requirement in the evaluation repository is satisfied, where the code lives, and how to reproduce the number. Numbers marked "measured" come from a command in this repo; numbers marked "report after run" require the voice stack (GPU, Deepgram, Cartesia keys) and are produced by the listed command on the submitter's machine.

## Problem 1: Self-correcting RAG

| Requirement | Where | How to verify |
|---|---|---|
| Baseline RAG (embed, top-k, answer) | `rag/baseline.py` | `make eval-rag` prints baseline column |
| Self-correcting loop: analyse, retrieve, evaluate evidence, rewrite, decompose, clarify, abstain | `rag/self_correcting.py`, `rag/state_machine.py`, `rag/query_rewriter.py`, `rag/evidence_evaluator.py` | same |
| Conflict detection between returns_policy_v1 and v2 | `rag/conflict_detector.py`; corpus `domains/ecommerce/knowledge_base/returns_policy_v{1,2}.md` | `DOMAIN=ecommerce make ingest eval-rag`, cases Q1, E005 to E009 |
| Evaluator questions Q1 to Q4 verbatim | `domains/ecommerce/eval/questions.json` (original) and mapped into `rag_eval.json` Q1 to Q4 | same |
| Hallucination rate baseline vs self-correcting | `eval/run_rag_eval.py` (forbidden-claim and fact checks per case) | summary JSON in `results/rag_eval_summary_<name>.json` |
| Latency P50/P95 per query | same harness, `latency` section | same |
| Average retrieval iterations | same harness | same |
| Abstention rate and clarification behaviour | same harness | same |
| Eval set size | BFSI 110 development + 100 holdout; e-commerce 40 | files under `domains/*/eval/` |
| Hallucination judging method | `docs/EVALUATION.md`, section "How hallucination is scored" | |

## Problem 3: Interruption handling

| Requirement | Where | How to verify |
|---|---|---|
| Interruption state machine (Round 1) | `docs/ARCHITECTURE.md`, section "Interruption state machine" | |
| Barge-in detection while speaking | `voice/runtime.py` `_speech_start`, `_barge_can_confirm`, `_confirm_barge_in` | logs `REAL BARGE-IN CONFIRMED` |
| Stop latency < 500 ms, measured end to end | `voice/barge_metrics.py`, client `barge_ack` in `web/index.html` | `make bench` barge-in table; report after run |
| False positive handling (noise, cough, TTS bleed) | `BARGE_MIN_SPEECH_MS`, ASR confirmation, browser `echoCancellation` | `make bench` shows suppressed candidates by reason |
| Context preserved after interruption | `voice/turn_manager.py`, `voice/runtime.py` generation ids; pending verification request survives | regression `scripts/test_identity_session_flow.py` |
| Stale transcript rejection | `voice/runtime.py` `on_transcript` | unit `tests/test_voice_turn_manager.py` |
| Three demo scenarios on orders data | `demo/SCRIPT.md`, `domains/ecommerce/data/orders.json` | demo video |
| Turn completion rate, coherence | `demo/SCRIPT.md` scripted 30-turn run; report after run | |

## Problem 4: Low-latency multilingual

| Requirement | Where | How to verify |
|---|---|---|
| Automatic language detection, no selector | `voice/language_detect.py`; Deepgram word tags in `voice/stt/deepgram.py`; UI defaults to Auto | `make eval-lid` (measured: 54 cases, 100%) |
| Evaluator utterances U1 to U4 | `domains/ecommerce/eval/utterances.json` and first four rows of `eval/datasets/language_detection_eval.json` | same |
| Reply in detected language; mixed resolves to Hindi | `VoiceSession._resolve_output_language` | unit `tests/test_language_detect.py` |
| Mid-session switch without reset | `VoiceSession._apply_output_language` keeps all session state | demo video segment 3 |
| Per-stage latency: STT, LLM, TTS, E2E, P50/P95 | `voice/turn_metrics.py`, `bench/latency_report.py` | `make bench`; report after run |
| Targets STT < 1.5 s, LLM first token < 2 s, TTS first audio < 1 s, E2E < 4 s | `bench/latency_report.py` compares P95 to targets | same |
| Language detection latency | `make eval-lid` prints mean detector time (measured: about 4 µs) | |
| Response-language correctness | `make eval-lid` | measured 100% |

## Problem 6: Guaranteed-delivery pipeline

| Requirement | Where | How to verify |
|---|---|---|
| POST /v1/jobs with Idempotency-Key, GET /v1/jobs/{id} in the stated shape | `jobs/api.py` | `tests/test_jobs_pipeline.py::test_api_contract` |
| Status lifecycle SUBMITTED → QUEUED → PROCESSING → COMPLETED / FAILED → RETRYING / DLQ | `jobs/store.py` `JobStatus`, `job_events` table | `GET /v1/jobs/{id}` returns `events` |
| Retries with exponential backoff and jitter, max attempts | `RetryPolicy` | `tests/chaos/permanent_failure_dlq.sh` prints backoff delays |
| Dead-letter queue with error context | `JobStore.fail`, `GET /v1/jobs/dlq`, `POST /v1/jobs/{id}/requeue` | same |
| Worker crash mid-job | lease-based claims, `JobStore.claim` reclaims expired leases | `tests/chaos/kill_worker_mid_job.sh` (SIGKILL a real worker process) |
| Broker restart | the SQLite WAL file is the queue; reopening it is the restart | `tests/chaos/restart_store.sh` |
| Duplicate submission | UNIQUE(idempotency_key) | `tests/chaos/duplicate_submit.sh` |
| Duplicate delivery / partial failure after side effect | `SideEffectLedger.once` | `tests/test_jobs_pipeline.py::test_duplicate_delivery_after_retry_single_side_effect` |
| Throughput | `tests/test_jobs_pipeline.py::test_throughput_is_measurable` writes `results/jobs_throughput.json` | measured about 250k jobs/min single worker, trivial handlers |
| Custom queue justification | `jobs/__init__.py` docstring; `docs/ARCHITECTURE.md` | |
| Wiring into the agent | `business/durable_executor.py`; enabled in `app/web_server.py` via `JOBS_ENABLED` | `tests/test_durable_executor.py` |

## Cross-cutting submission requirements

| Requirement | Where |
|---|---|
| README: setup, run, tests, benchmarks, assumptions, tradeoffs, limitations, AI tools used | `README.md` |
| Docker Compose | `docker-compose.yml`, `Dockerfile` |
| No secrets | `.env.example`, `.gitignore` |
| Round 1: PRD, architecture, solution overview, success metrics | `docs/PRD.md`, `docs/ARCHITECTURE.md`, `docs/SOLUTION_OVERVIEW.md`, `docs/PROPOSAL.md` |
| Demo | `demo/SCRIPT.md` |
| Evaluation honesty | `docs/EVALUATION.md`, "Which numbers are blind" |

## Not covered, by decision

| Item | Reason |
|---|---|
| Problem 2 document extraction | Different product; out of scope in `docs/PROPOSAL.md` |
| Problem 5 gateway | Listed as future work; the agent already abstracts three providers behind small clients so failover is a contained addition |
| CSM-1B TTS fine-tune | Cartesia Hindi chosen for latency and quality; training a TTS model is a separate project |
| Router retrained for e-commerce intents | The e-commerce pack covers RAG, language and evaluation; routing and tools stay BFSI |
