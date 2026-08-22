# Changelog

## 1.0.0 (submission)

### Added
- Automatic per-utterance language detection (`voice/language_detect.py`), Deepgram word-language hints, Auto default in the UI, mid-session switching without state reset, 54-case eval (`make eval-lid`).
- End-to-end barge-in stop-latency measurement with client ack, minimum voiced duration, suppressed-candidate counting (`voice/barge_metrics.py`).
- Per-turn stage latency records and the Problem 4 report (`voice/turn_metrics.py`, `bench/latency_report.py`, `make bench`).
- Domain packs: `domains/bfsi` (moved) and `domains/ecommerce` with the evaluators' data verbatim plus a 40-case RAG eval; `DOMAIN` env switch (`rag/config.py`).
- Guaranteed-delivery job pipeline (`jobs/`): SQLite WAL store, idempotency, leases, backoff, DLQ, side-effect ledger, worker, REST API, 9 tests, 4 chaos scripts.
- Durable write decorator wiring business writes into jobs (`business/durable_executor.py`), on by default via `JOBS_ENABLED`.
- Docker Compose (GPU and hosted profiles), Dockerfile, Makefile, pinned requirements.
- Docs: PROPOSAL (PS7 template), PRD, COVERAGE, SOLUTION_OVERVIEW, interruption state machine and job lifecycle in ARCHITECTURE, evaluation honesty and hallucination scoring in EVALUATION, demo script and scripted turns.

### Changed
- README restructured around the submission rubric; evaluation claims labelled as locked, development or blind.
- `eval/run_rag_eval.py` and benchmark scripts default to the active domain pack.

### Removed
- Dead modules `core/`, `audio/`, `tools/` (not imported anywhere).
