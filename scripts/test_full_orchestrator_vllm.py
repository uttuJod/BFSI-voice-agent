from __future__ import annotations

import asyncio
import json

from business import (
    BusinessDatabase,
    BusinessRepository,
    BusinessToolExecutor,
)
from integration.orchestrator import BFSIOrchestrator
from integration.production_router import ProductionBFSIRouter
from integration.runtime_latency import install_latency_optimizations


async def main() -> None:
    db = BusinessDatabase("results/production_integration_test.sqlite3")
    db.initialize()
    db.seed_demo_data()

    repo = BusinessRepository(db)
    tools = BusinessToolExecutor(repo)

    router = ProductionBFSIRouter(
        base_url="http://127.0.0.1:8001/v1",
        model="bfsi-router",
        timeout_seconds=30.0,
    )

    orchestrator = BFSIOrchestrator(
        router=router,
        tool_executor=tools,
    )

    install_latency_optimizations(
        orchestrator,
        move_embeddings_cpu=True,
    )

    tests = [
        "What is my outstanding balance?",
        "I already paid yesterday.",
        "What is the grace period policy?",
        "Please verify me before retrieving my account details.",
        "I might manage ₹1900 sometime next week.",
        "I will pay ₹1900 on 25 August 2026.",
    ]

    try:
        for text in tests:
            result = await orchestrator.handle(
                text,
                customer_id="CUST-1001",
            )

            print("\nUSER:", text)
            print(
                json.dumps(
                    result.model_dump(),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

        print(
            "\nLATENCY:",
            json.dumps(
                orchestrator.save_latency_metrics(),
                indent=2,
            ),
        )
    finally:
        router.close()


if __name__ == "__main__":
    asyncio.run(main())
