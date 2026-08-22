from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from business import (
    BusinessDatabase,
    BusinessRepository,
    BusinessToolExecutor,
)

from integration import (
    BFSIOrchestrator,
    QwenBFSIRouter,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parent


ADAPTER_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen35_customer_support_FINAL_FROZEN"
)


DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "business.db"
)


SESSION_CUSTOMER_ID = (
    os.getenv(
        "BFSI_CUSTOMER_ID",
        "CUST-1001",
    )
)


async def main():

    print(
        "=" * 70
    )

    print(
        "BFSI QWEN + RAG + BUSINESS TOOLS"
    )

    print(
        "=" * 70
    )

    # ============================================================
    # DATABASE
    # ============================================================

    db = BusinessDatabase(
        DB_PATH
    )

    db.initialize()

    #
    # Development/demo only.
    #
    db.seed_demo_data()

    repository = (
        BusinessRepository(
            db
        )
    )

    tool_executor = (
        BusinessToolExecutor(
            repository
        )
    )

    print(
        f"[DB] {DB_PATH}"
    )

    print(
        "[SESSION] Customer:",
        SESSION_CUSTOMER_ID,
    )

    # ============================================================
    # QWEN
    # ============================================================

    router = QwenBFSIRouter(
        adapter_path=(
            ADAPTER_PATH
        ),
    )

    # ============================================================
    # ORCHESTRATOR
    # ============================================================

    orchestrator = (
        BFSIOrchestrator(
            router=router,
            tool_executor=(
                tool_executor
            ),
        )
    )

    print()

    print(
        "Ready. Type 'exit' to quit."
    )

    print()

    # ============================================================
    # CLI LOOP
    # ============================================================

    while True:

        user_text = input(
            "USER > "
        ).strip()

        if not user_text:
            continue

        if user_text.lower() in {
            "exit",
            "quit",
        }:
            break

        result = (
            await orchestrator.handle(
                user_text=(
                    user_text
                ),
                customer_id=(
                    SESSION_CUSTOMER_ID
                ),
            )
        )

        print()

        print(
            "ASSISTANT >",
            result.final_response,
        )

        print()

        print(
            "--- ROUTING TRACE ---"
        )

        trace = {
            "intent":
                result.router.intent,

            "confidence":
                result.router.confidence,

            "requires_rag":
                result.router
                .requires_rag,

            "requires_tool":
                result.router
                .requires_tool,

            "selected_tool":
                result.router.tool,

            "tool_arguments":
                result.router.arguments,

            "source":
                result.source,

            "rag_used":
                result.rag_used,

            "rag_verdict":
                result.rag_verdict,

            "rag_citations":
                result.rag_citations,

            "tool_executed":
                result.tool_executed,

            "tool_success":
                result.tool_success,

            "tool_status":
                result.tool_status,

            "tool_name":
                result.tool_name,

            "tool_result":
                result.tool_result,

            "tool_error_code":
                result.tool_error_code,

            "guard_actions":
                result.guard_actions,
        }

        print(
            json.dumps(
                trace,
                indent=2,
                ensure_ascii=False,
            )
        )

        print()


if __name__ == "__main__":

    asyncio.run(
        main()
    )