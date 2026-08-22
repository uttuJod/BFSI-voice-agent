from __future__ import annotations

import asyncio
import time

from app.web_server import (
    _build_orchestrator,
)


async def main() -> None:
    orchestrator = (
        _build_orchestrator()
    )

    queries = [
        "What is my outstanding balance?",
        "I already paid.",
        "What is my grace period?",
    ]

    for query in queries:
        started = (
            time.perf_counter()
        )

        result = await orchestrator.handle(
            user_text=query,
            customer_id="CUST-1001",
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        print()
        print("USER:", query)
        print(
            "INTENT:",
            result.router.intent,
        )
        print(
            "TIME:",
            round(
                elapsed,
                2,
            ),
            "seconds",
        )
        print(
            "RESPONSE:",
            result.final_response,
        )


if __name__ == "__main__":
    asyncio.run(
        main()
    )
