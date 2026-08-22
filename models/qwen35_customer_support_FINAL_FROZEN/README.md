---
base_model: unsloth/Qwen3.5-4B
library_name: peft
pipeline_tag: text-generation
tags:
  - qwen3.5
  - lora
  - sft
  - bfsI
  - intent-classification
  - tool-routing
  - multilingual
  - english
  - hindi
  - hinglish
  - vllm
---

# Qwen3.5-4B BFSI Routing Adapter

Fine-tuned LoRA adapter for the routing layer of the **BFSI Voice Agent**.

The model converts a customer utterance into a strict structured decision used by the production orchestrator. It is responsible for **intent classification, tool selection, RAG routing, clarification decisions, and structured arguments**. It does not directly execute tools or access customer data.

## Base model

- **Base model:** `unsloth/Qwen3.5-4B`
- **Fine-tuning method:** supervised fine-tuning with LoRA
- **PEFT type:** LoRA
- **LoRA rank:** 32
- **LoRA alpha:** 32
- **RS-LoRA:** enabled
- **LoRA dropout:** 0.0
- **Task type:** causal language modeling

The checked-in `adapter_config.json` contains the exact PEFT configuration used by this adapter.

## Purpose

This router is specialized for multilingual BFSI collections and customer-support conversations. It is designed to recognize English, Hindi, and Hinglish requests and produce exactly one JSON decision matching the application schema.

The router is deliberately separated from business execution. Sensitive actions remain behind deterministic application code and verification controls.

## Supported intents

The router predicts one of 17 intents:

1. `payment_reminder`
2. `promise_to_pay`
3. `callback_request`
4. `paid_already`
5. `partial_payment`
6. `financial_hardship`
7. `dispute`
8. `wrong_number`
9. `refusal_to_pay`
10. `human_escalation`
11. `policy_question`
12. `account_status`
13. `identity_verification`
14. `ambiguous`
15. `out_of_scope`
16. `privacy_sensitive`
17. `prompt_injection`

## Supported tool selections

The model may select one of these tools for the orchestrator:

- `get_customer_account`
- `get_outstanding_balance`
- `record_promise_to_pay`
- `schedule_callback`
- `record_payment_reported`
- `open_dispute`
- `record_financial_hardship`
- `request_human_escalation`
- `get_policy`
- `get_call_history`

The model only selects tools. It never executes them directly.

## Structured output

Each prediction conforms to the application `SupportDecision` schema and contains fields for:

- intent
- confidence
- whether RAG is required
- whether a tool is required
- selected tool
- tool arguments
- response style
- clarification requirement
- concise response guidance

Production vLLM inference uses structured output validation to keep responses schema-conformant.

## Frozen inference configuration

The production router is evaluated with deterministic generation:

- `enable_thinking=False`
- chat template applied with `add_generation_prompt=True`
- `add_special_tokens=False`
- maximum output tokens: `320`
- sampling disabled
- cache enabled
- vLLM structured output enabled

The exact system prompt used by the current router lives in `integration/vllm_router.py`.

## Evaluation

### Historical frozen artifact

The original `final_evaluation.json` stored with this adapter records the Stage-2B checkpoint at:

| Metric | Result |
|---|---:|
| Cases | 120 |
| JSON valid | 100% |
| Schema valid | 100% |
| Intent accuracy | 97.5% |
| Tool accuracy | 97.5% |
| RAG routing accuracy | 100% |
| Clarification accuracy | 100% |
| Argument accuracy | 97.5% |
| Passed cases | 117 / 120 |

That file is kept as a historical frozen artifact and is not rewritten.

### Current vLLM reproduction

Using the current production vLLM path with the same frozen adapter and deterministic prompt/template settings, the locked 120-case benchmark reproduces:

| Metric | Result |
|---|---:|
| Cases | 120 |
| JSON valid | 100% |
| Schema valid | 100% |
| Intent accuracy | 98.33% |
| Tool accuracy | 98.33% |
| RAG routing accuracy | 100% |
| Clarification accuracy | 100% |
| Argument accuracy | 98.33% |
| Passed cases | 118 / 120 |

The two remaining model-level misses are both own-account identity-verification taxonomy boundaries. Production orchestration applies a narrow deterministic normalization guard for those cases without modifying the router prompt or model output distribution.

## Safety model

This adapter is not the security boundary of the application.

The production system adds deterministic controls around the router for:

- identity verification before protected account access
- third-party privacy requests
- unsafe credential requests
- vague promise-to-pay dates
- business-tool argument validation
- prompt-injection handling
- current-policy RAG selection

Dynamic account facts come from business tools, not model memory. Policy answers come from the RAG system, not unsupported generation.

## Deployment

The production path serves the base model with this LoRA adapter through vLLM using the OpenAI-compatible API.

Example launcher:

```bash
./wsl/start_vllm.sh
```

The application expects the LoRA model name `bfsi-router` at the configured vLLM endpoint.

## Repository contents

This directory intentionally contains configuration and reproducibility metadata but **does not include the LoRA weight file** in the public repository.

Included files may contain:

- `adapter_config.json`
- `chat_template.jinja`
- tokenizer/config metadata
- `final_evaluation.json`
- this model card

The actual `adapter_model.safetensors` is excluded from Git tracking.

## Limitations

- This is a task-specific routing adapter, not a general-purpose banking assistant.
- Benchmark accuracy does not replace deterministic authorization or privacy enforcement.
- The local demo identity-verification mechanism is not bank-grade MFA/KYC.
- Production deployment would require approved authentication, secure secret management, audit logging, rate limiting, monitoring, and regulatory review.
- The system should be tested with synthetic/demo customer data only.

## Related project documentation

- Root `README.md` — complete BFSI Voice Agent overview
- `docs/ARCHITECTURE.md` — system architecture
- `docs/EVALUATION.md` — evaluation and regression coverage
- `docs/SECURITY.md` — security model and deployment limitations
