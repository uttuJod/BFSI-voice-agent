from __future__ import annotations

import asyncio

from rag.self_correcting import SelfCorrectingRAG


async def main() -> None:
    rag = SelfCorrectingRAG()

    result = await rag.answer(
        question="What is the grace period policy?",
        context={
            "intent": "policy_question",
            "language": "en",
        },
    )

    print("answerable:", result.answerable)
    print("verdict:", result.verdict)
    print("answer:", result.answer)
    print("citations:", result.citations)

    answer = (
        result.answer
        or ""
    ).lower()

    citation_doc_ids = {
        str(
            getattr(
                citation,
                "document_id",
                "",
            )
        )
        for citation in result.citations
    }

    assert result.answerable
    assert "7 days" in answer
    assert "5 days" not in answer

    assert (
        "collections_policy_v2"
        in citation_doc_ids
    )

    assert (
        "collections_policy"
        not in citation_doc_ids
    )

    print()
    print("RAG ACTIVE-VERSION FILTER: PASS")


if __name__ == "__main__":
    asyncio.run(main())
