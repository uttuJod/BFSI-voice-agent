from integration.orchestrator_vllm import BFSIOrchestrator


TESTS = [
    "What is my outstanding balance?",
    "I already paid yesterday.",
    "What is the grace period policy?",
    "Please verify me before retrieving my account details.",
    "I might manage ₹1900 sometime next week.",
]


def main():
    orch = BFSIOrchestrator()

    print("HEALTH:", orch.health())
    print()

    try:
        for text in TESTS:
            result = orch.handle(text)

            print("USER:", text)
            print("INTENT:", result.intent)
            print("TOOL:", result.tool)
            print("RAG:", result.requires_rag)
            print("CLARIFY:", result.needs_clarification)
            print("FINAL:", result.final_response)
            print("-" * 80)

        print("LATENCY:", orch.save_latency_metrics())

    finally:
        orch.close()


if __name__ == "__main__":
    main()
