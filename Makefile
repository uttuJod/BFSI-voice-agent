.PHONY: setup test chaos eval eval-rag eval-lid bench router-bench ingest run worker jobs-api docker

PY ?= python3

setup:            ## install app dependencies (vLLM lives in wsl/install_vllm.sh)
	$(PY) -m pip install -r requirements.txt

test:             ## unit + integration tests that do not need a GPU or API keys
	$(PY) -m pytest tests -q --ignore=tests/test_vllm_router_contract.py

test-all:         ## everything, including the live vLLM contract test
	$(PY) -m pytest tests -q

chaos:            ## guaranteed-delivery failure tests against real worker processes
	bash tests/chaos/run_all.sh

ingest:           ## build the FAISS index for the active DOMAIN (default bfsi)
	$(PY) -m rag.ingestion

eval-rag:         ## baseline vs self-correcting RAG on domains/$(DOMAIN)/eval/rag_eval.json
	$(PY) -m eval.run_rag_eval --name $${DOMAIN:-bfsi}

eval-rag-blind:   ## run ONCE on the untouched blind set, then never tune against it
	$(PY) -m eval.run_rag_eval --dataset domains/bfsi/eval/rag_eval_holdout.json --name blind

eval-lid:         ## language detection accuracy
	$(PY) -m eval.run_language_detection_eval

router-bench:     ## locked 120-case router benchmark through vLLM
	$(PY) -m scripts.run_vllm_120_benchmark

eval: eval-lid eval-rag router-bench ## all offline evaluations

bench:            ## per-stage voice latency table from recorded sessions
	$(PY) -m bench.latency_report --markdown

run:              ## start the voice agent (expects vLLM on VLLM_BASE_URL)
	$(PY) -m app.main

worker:           ## start a guaranteed-delivery worker
	$(PY) -m jobs.worker

jobs-api:         ## jobs REST API on :8010
	uvicorn jobs.api:app --port 8010

docker:
	docker compose up --build

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'
