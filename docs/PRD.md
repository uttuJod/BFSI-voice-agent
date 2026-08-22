# Product Requirements: BFSI Voice Agent

## 1. Purpose

Give a lender a voice agent that can run first-contact collections and support calls in English, Hindi and Hinglish, with the same guarantees a supervised human agent is held to: verify before disclosing, answer policy only from the policy, record every commitment once, and never guess a date or an amount.

## 2. Users and jobs to be done

| User | Job | What "done" looks like |
|---|---|---|
| Borrower | Understand my balance, make a promise, ask a rule, dispute a record, ask for a human | Heard in their language, answered within a conversational pause, not asked to repeat themselves after an interruption |
| Collections ops | Capture promises, callbacks, disputes, hardship flags | Each appears exactly once in the system of record with a trace of how it was captured |
| Compliance | Prove the agent cannot leak data or fabricate policy | Verification gate, privacy guard and grounding are enforced in code and covered by regression tests |

## 3. Functional requirements

### 3.1 Conversation
- FR1. Accept live microphone audio from a browser and stream a spoken reply.
- FR2. Detect end of user speech within 550 ms of silence in normal turns and 1600 ms during digit entry.
- FR3. Allow the user to interrupt the agent at any point; stop audio within 500 ms; keep the interrupted context.
- FR4. Ignore late or duplicate STT results after a turn has been accepted.

### 3.2 Language
- FR5. Recognise English, Hindi and code-mixed Hinglish input without a language selector.
- FR6. Reply in the language of the current utterance by default (Hinglish resolves to Hindi). Allow a fixed English or Hindi override.
- FR7. Switch language mid-session without losing verification, pending requests or idempotency state.
- FR8. Speak amounts and dates in the reply language's spoken form.

### 3.3 Understanding and safety
- FR9. Classify each utterance into one of 17 intents and select at most one business tool, as a validated JSON decision.
- FR10. Block account disclosure until the session is verified; resume the original request automatically after verification.
- FR11. Refuse third-party account requests and never request or reveal OTP, CVV, PIN, password or full card number.
- FR12. Ask for clarification on vague dates; reject amounts above the known balance.

### 3.4 Knowledge
- FR13. Answer policy questions only from the versioned knowledge base with citations.
- FR14. Detect conflicts between policy versions, prefer the active version for present-tense questions, and keep superseded versions for historical questions.
- FR15. Abstain when evidence is insufficient or the question is out of scope; ask for clarification when ambiguous.

### 3.5 Actions
- FR16. Record promise to pay, callback, payment claim, dispute, hardship and escalation.
- FR17. Deliver downstream side effects exactly once under worker crash, duplicate submission and store restart, with retries, backoff and a dead-letter queue.
- FR18. Confirm to the caller once the action is durably queued, not after downstream completion.

### 3.6 Operability
- FR19. Record per-stage latency for every turn and aggregate into P50/P95 tables.
- FR20. Run every evaluation and benchmark from a single command with reproducible output files.
- FR21. Start the full stack with Docker Compose; support a hosted OpenAI-compatible LLM when no GPU is present.

## 4. Non-functional requirements

| Area | Requirement |
|---|---|
| Latency | Speech end to first audio P50 under 2.5 s, P95 under 4 s; router P95 under 2 s; TTS first audio P95 under 1 s |
| Accuracy | Router intent accuracy at least 95% on the locked benchmark; RAG behaviour accuracy at least 90% on the blind set |
| Reliability | Zero duplicate side effects in chaos tests; no lost jobs on restart |
| Security | No secrets in the repository; synthetic customer data only; verification gate outside the LLM |
| Portability | Runs on WSL2 or Linux with an 8 GB GPU, or on CPU with a hosted model |

## 5. Assumptions

- One customer per voice session, identified by the session's customer id; the demo verification (last four digits of the registered mobile) stands in for bank-grade authentication.
- Deepgram and Cartesia are reachable with valid keys; latency figures assume a typical Indian broadband connection.
- Policy documents carry version and status metadata in front matter.
- The fine-tuned router is frozen; behavioural fixes are made with deterministic guards and tested, not by re-prompting.

## 6. Known failure modes and mitigations

| Failure mode | Mitigation |
|---|---|
| STT renders spoken digits as currency or merges them | Verification-specific normaliser and longer endpoint window |
| Late STT final arrives after the turn closed | Turn-state check drops stale transcripts |
| TTS bleed-through or a cough triggers barge-in | 250 ms minimum voiced duration, ASR confirmation, browser echo cancellation; suppressed candidates are counted |
| Router mislabels own-account verification phrasing | Deterministic normalisation guard covering the 2 known misses |
| Two policy versions retrieved together | Conflict detector plus active-version preference with citations |
| Worker dies holding a job | Lease expiry and reclaim by another worker; side-effect ledger prevents repeats |
| Same request retried by the client | Idempotency key on the job row; duplicate returns the original job |
| Language mis-detected on a short utterance | Devanagari is decisive; Hinglish needs lexicon evidence; STT hint only breaks ties; fixed override available |

## 7. Out of scope

Telephony ingress, KYC, production secret management, multi-tenant gateway, document extraction, TTS model training, retraining the router for other domains.
