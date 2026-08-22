# Demo script (8 minutes)

Record with the voice UI open, terminal logs visible, `DOMAIN` switched between segments.

| Time | Segment | What to show | Evidence on screen |
|---|---|---|---|
| 0:00 | Architecture | `docs/ARCHITECTURE.md` component diagram, one sentence per box | |
| 1:00 | E-commerce, interruption scenario 1 | Set `DOMAIN=ecommerce`. Say "I want to return order ORD123" and, while the agent responds, cut in with "Sorry, ORD124". Agent must continue with ORD124 | `REAL BARGE-IN CONFIRMED`, `BARGE \| stop_latency_ms=...` in log |
| 2:00 | Scenario 2 | Ask "What is your refund policy?", interrupt with "Just tell me the return window". Agent gives the window only, from returns_policy_v2 (7 days) | RAG citations in `ai_trace` |
| 2:45 | Scenario 3 | Ask "How long does shipping take?", interrupt mid-sentence with "Actually, I want to cancel". Agent pivots to cancellation and asks for the order id | |
| 3:30 | Language switch | Language selector on Auto. Speak English, then Hindi ("मेरा ऑर्डर कहाँ है?"), then Hinglish ("Order ORD124 cancel karna hai"), with no restart. Replies follow the caller's language | `LANGUAGE SWITCH \| english -> hindi \| session state retained` |
| 4:45 | BFSI | `DOMAIN=bfsi`. Ask for balance, get the verification challenge, say the digits, watch the original request resume. Ask the grace period (conflict between policy v1 and v2, answer from v2 with citation). Ask for an OTP, get refused | `IDENTITY VERIFIED`, conflict flag in trace, `credential_safety` guard |
| 6:00 | Durable write | Say "I will pay five thousand on the fifth". Show `DURABLE WRITE \| job_id=...`. In a second terminal run `bash tests/chaos/kill_worker_mid_job.sh` and show PASS | |
| 7:00 | Numbers | `make bench --markdown` output, `make eval-lid`, router benchmark summary, `results/rag_eval_summary_ecommerce.json` | |
| 7:45 | Limits | Say what is demo-grade (verification, single-host queue) and what is not covered (telephony, PS2, PS5) | |

## Scripted 30-turn run for turn-completion and coherence

`demo/turns.txt` lists 30 utterances (10 per language) with 10 planned interruptions. After the run, `make bench` gives the latency tables; coherence is scored 1 to 5 per interrupted turn by a reviewer reading the transcript, using one question: did the reply address the interruption without repeating the cancelled sentence? Record the mean in `docs/EVALUATION.md`.
