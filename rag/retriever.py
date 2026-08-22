from __future__ import annotations

from .config import RAGConfig
from .embeddings import EmbeddingModel
from .schemas import (
    RetrievedChunk,
    SearchFilters,
    SearchRequest,
)
from .vector_store import FAISSVectorStore


class Retriever:
    """
    Global semantic retrieval with optional soft metadata boosting.

    Domain prediction is NOT a hard filter anymore.
    """

    DOMAIN_BOOST = 0.08
    ACTIVE_BOOST = 0.015

    def __init__(
        self,
        config: RAGConfig,
    ):
        self.config = config

        self.embedder = EmbeddingModel(
            config.embedding_model
        )

        self.store = (
            FAISSVectorStore.load(
                config.index_dir
            )
        )

    def retrieve(
        self,
        request: SearchRequest,
    ) -> list[RetrievedChunk]:

        qv = self.embedder.encode(
            [request.query]
        )[0]

        soft_domain = (
            request.filters.domain
            if request.filters
            else None
        )

        hard_filters = None

        if request.filters:

            hard_filters = SearchFilters(
                domain=None,
                status=request.filters.status,
                document_ids=(
                    request.filters.document_ids
                ),
                effective_on_or_before=(
                    request.filters
                    .effective_on_or_before
                ),
                metadata=(
                    request.filters.metadata
                ),
            )

        candidate_k = min(
            max(
                request.top_k * 4,
                request.top_k + 8,
            ),
            max(
                1,
                len(self.store.chunks),
            ),
        )

        candidates = self.store.search(
            qv,
            top_k=candidate_k,
            score_threshold=(
                request.score_threshold
            ),
            filters=hard_filters,
        )

        if not candidates:
            return []

        def rerank_score(
            chunk: RetrievedChunk,
        ) -> float:

            score = float(
                chunk.score
            )

            if (
                soft_domain
                and chunk.metadata.domain
                == soft_domain
            ):
                score += self.DOMAIN_BOOST

            status = getattr(
                chunk.metadata.status,
                "value",
                chunk.metadata.status,
            )

            if status == "active":
                score += self.ACTIVE_BOOST

            return score

        ranked = sorted(
            candidates,
            key=rerank_score,
            reverse=True,
        )[: request.top_k]

        return [
            chunk.model_copy(
                update={
                    "rank": rank,
                }
            )
            for rank, chunk
            in enumerate(
                ranked,
                start=1,
            )
        ]

    def retrieve_many(
        self,
        requests: list[SearchRequest],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:

        if not requests:
            return []

        merged: dict[
            str,
            RetrievedChunk,
        ] = {}

        for request in requests:

            for chunk in self.retrieve(
                request
            ):

                previous = merged.get(
                    chunk.chunk_id
                )

                if (
                    previous is None
                    or float(chunk.score)
                    > float(previous.score)
                ):
                    merged[
                        chunk.chunk_id
                    ] = chunk

        limit = (
            top_k
            or self.config.top_k
        )

        ranked = sorted(
            merged.values(),
            key=lambda chunk: (
                float(chunk.score)
            ),
            reverse=True,
        )[: max(
            limit,
            self.config.top_k,
        )]

        return [
            chunk.model_copy(
                update={
                    "rank": rank,
                }
            )
            for rank, chunk
            in enumerate(
                ranked,
                start=1,
            )
        ]