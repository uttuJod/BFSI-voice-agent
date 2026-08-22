import time
from .config import RAGConfig
from .retriever import Retriever
from .answer_generator import GroundedAnswerGenerator
from .schemas import (
    RAGResult, SearchRequest, EvidenceVerdict, LatencyBreakdown,
)

class BaselineRAG:
    def __init__(self, config=None):
        self.config = config or RAGConfig()
        self.retriever = Retriever(self.config)
        self.generator = GroundedAnswerGenerator()

    async def answer(self, question: str, context=None):
        start = time.perf_counter()
        r0 = time.perf_counter()
        chunks = self.retriever.retrieve(SearchRequest(
            query=question,
            top_k=self.config.top_k,
            score_threshold=self.config.score_threshold,
        ))
        retrieval_ms = (time.perf_counter() - r0) * 1000

        if not chunks:
            total = (time.perf_counter() - start) * 1000
            return RAGResult(
                answer=None,
                answerable=False,
                confidence=0.8,
                verdict=EvidenceVerdict.INSUFFICIENT,
                retrieval_iterations=1,
                query_history=[question],
                abstention_reason="No sufficiently relevant evidence retrieved.",
                latency=LatencyBreakdown(retrieval_ms=retrieval_ms, total_ms=total),
            )

        g0 = time.perf_counter()
        answer, citations = self.generator.generate(question, chunks)
        generation_ms = (time.perf_counter() - g0) * 1000
        total = (time.perf_counter() - start) * 1000

        return RAGResult(
            answer=answer,
            answerable=True,
            confidence=max(c.score for c in chunks),
            verdict=EvidenceVerdict.SUFFICIENT,
            retrieval_iterations=1,
            query_history=[question],
            retrieved_chunks=chunks,
            citations=citations,
            latency=LatencyBreakdown(
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                total_ms=total,
            ),
        )
