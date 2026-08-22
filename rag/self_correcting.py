from __future__ import annotations

import re
import time

from .answer_generator import GroundedAnswerGenerator
from .config import RAGConfig
from .conflict_detector import ConflictDetector
from .evidence_evaluator import EvidenceEvaluator
from .query_analyzer import QueryAnalyzer
from .query_rewriter import QueryRewriter
from .retriever import Retriever
from .schemas import (
    ConflictResolutionStatus,
    CorrectionAction,
    EvidenceVerdict,
    LatencyBreakdown,
    PipelineState,
    RAGContext,
    RAGResult,
    RetrievalTrace,
    SearchFilters,
    SearchRequest,
)
from .tracing import TraceRecorder


class SelfCorrectingRAG:

    def __init__(self, config=None):

        self.config = config or RAGConfig()

        self.analyzer = QueryAnalyzer()

        self.retriever = Retriever(
            self.config
        )

        self.evaluator = EvidenceEvaluator(
            self.config.low_relevance_threshold,
            self.config.sufficient_threshold,
            self.config.strong_relevance_threshold,
        )

        self.conflicts = ConflictDetector()

        self.rewriter = QueryRewriter()

        self.generator = GroundedAnswerGenerator()

    # ============================================================
    # SEARCH
    # ============================================================

    def _search_request(
        self,
        query: str,
        analysis,
        top_k: int | None = None,
    ):

        filters = (
            SearchFilters(
                domain=analysis.domain
            )
            if analysis.domain
            else None
        )

        return SearchRequest(
            query=query,
            top_k=(
                top_k
                or self.config.top_k
            ),
            score_threshold=(
                self.config.score_threshold
            ),
            filters=filters,
        )

    # ============================================================
    # MULTI-PART RETRIEVAL
    # ============================================================

    def _retrieve_for_analysis(
        self,
        query: str,
        analysis,
    ):

        if (
            analysis.query_type.value
            == "multi_part"
            and analysis.subqueries
        ):

            requests = []

            for subquery in analysis.subqueries:

                sub_analysis = (
                    self.analyzer.analyze(
                        subquery
                    )
                )

                requests.append(
                    self._search_request(
                        subquery,
                        sub_analysis,
                    )
                )

            chunks = (
                self.retriever.retrieve_many(
                    requests,
                    top_k=max(
                        self.config.top_k,
                        min(
                            10,
                            len(requests) * 4,
                        ),
                    ),
                )
            )

            return (
                chunks,
                CorrectionAction.QUERY_DECOMPOSITION,
            )

        return (
            self.retriever.retrieve(
                self._search_request(
                    query,
                    analysis,
                )
            ),
            CorrectionAction.NONE,
        )

    # ============================================================
    # EVIDENCE MERGING
    # ============================================================

    @staticmethod
    def _merge_chunks(
        existing,
        new_chunks,
    ):
        """
        Preserve useful evidence across retrieval iterations.
        """

        merged = {}

        for chunk in (
            list(existing)
            + list(new_chunks)
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

        ranked = sorted(
            merged.values(),
            key=lambda chunk: float(
                chunk.score
            ),
            reverse=True,
        )

        return ranked[:10]

    # ============================================================
    # RELEVANT EVIDENCE
    # ============================================================

    @staticmethod
    def _relevant_chunks(
        chunks,
        supported_document_ids,
    ):

        if not supported_document_ids:
            return chunks

        supported = set(
            supported_document_ids
        )

        narrowed = [
            chunk
            for chunk in chunks
            if chunk.document_id
            in supported
        ]

        return narrowed or chunks

    # ============================================================
    # AUTHORITATIVE / ACTIVE POLICY FILTER
    # ============================================================

    @staticmethod
    def _prefer_active_policy_versions(
        question: str,
        chunks,
    ):
        """
        Prevent a normal/current policy answer from mixing an active
        policy with a superseded version of the same policy family.

        Important:
        - Retrieval is NOT hard-filtered. Old evidence is still available
          for conflict detection and historical/version-specific questions.
        - Only the final answer-generation evidence is narrowed.
        - Explicit historical questions keep superseded evidence.
        """

        if not chunks:
            return chunks

        q = re.sub(
            r"\s+",
            " ",
            str(question).lower().strip(),
        )

        historical_terms = (
            "previous",
            "old policy",
            "older policy",
            "superseded",
            "historical",
            "earlier version",
            "previous version",
            "version 1",
            "v1",
            "five-day",
            "five day",
            "5-day",
            "5 day",
        )

        if any(
            term in q
            for term in historical_terms
        ):
            return chunks

        def family_key(chunk):
            metadata = getattr(
                chunk,
                "metadata",
                None,
            )

            title = str(
                getattr(
                    metadata,
                    "title",
                    "",
                )
                or ""
            ).strip().lower()

            if title:
                return (
                    "title",
                    re.sub(
                        r"\s+",
                        " ",
                        title,
                    ),
                )

            document_id = str(
                getattr(
                    chunk,
                    "document_id",
                    "",
                )
                or ""
            ).lower()

            document_id = re.sub(
                r"(?:[_-]?v(?:ersion)?[_-]?\d+)$",
                "",
                document_id,
            )

            return (
                "document_id",
                document_id,
            )

        families = {}

        for chunk in chunks:
            families.setdefault(
                family_key(chunk),
                [],
            ).append(chunk)

        kept = []

        for family_chunks in families.values():
            active = [
                chunk
                for chunk in family_chunks
                if str(
                    getattr(
                        getattr(
                            chunk,
                            "metadata",
                            None,
                        ),
                        "status",
                        "",
                    )
                ).lower().endswith(
                    "active"
                )
            ]

            if not active:
                kept.extend(
                    family_chunks
                )
                continue

            #
            # If the family has an active version, never let a superseded
            # member of that same family enter the normal answer.
            #
            candidates = [
                chunk
                for chunk in family_chunks
                if not str(
                    getattr(
                        getattr(
                            chunk,
                            "metadata",
                            None,
                        ),
                        "status",
                        "",
                    )
                ).lower().endswith(
                    "superseded"
                )
            ]

            #
            # Multiple active versions: prefer the latest metadata version
            # for generation, but preserve non-versioned sibling evidence.
            #
            active_versions = [
                (
                    getattr(
                        chunk.metadata,
                        "effective_date",
                        None,
                    ),
                    getattr(
                        chunk.metadata,
                        "version",
                        None,
                    ),
                    chunk,
                )
                for chunk in candidates
                if str(
                    getattr(
                        chunk.metadata,
                        "status",
                        "",
                    )
                ).lower().endswith(
                    "active"
                )
            ]

            if len(active_versions) > 1:
                active_versions.sort(
                    key=lambda item: (
                        item[0]
                        or __import__(
                            "datetime"
                        ).date.min,
                        item[1]
                        or 0,
                    ),
                    reverse=True,
                )

                preferred_doc = (
                    active_versions[0][2]
                    .document_id
                )

                candidates = [
                    chunk
                    for chunk in candidates
                    if (
                        not str(
                            getattr(
                                chunk.metadata,
                                "status",
                                "",
                            )
                        ).lower().endswith(
                            "active"
                        )
                        or chunk.document_id
                        == preferred_doc
                    )
                ]

            kept.extend(
                candidates
            )

        kept_ids = {
            chunk.chunk_id
            for chunk in kept
        }

        ordered = [
            chunk
            for chunk in chunks
            if chunk.chunk_id
            in kept_ids
        ]

        return ordered or chunks

    # ============================================================
    # POLICY CONFLICT TRIGGER
    # ============================================================

    @staticmethod
    def _query_can_contain_policy_conflict(
        question: str,
        analysis,
    ) -> bool:
        """
        Decide whether explicit policy-version conflict handling
        should run.

        Examples that SHOULD trigger:
            - latest collections rule
            - current grace-period policy
            - previous five-day rule
            - which version applies?
            - documents disagree
            - which policy takes precedence?

        Examples that should NOT trigger:
            - mark my account fully current
            - is my payment current?
        """

        q = re.sub(
            r"\s+",
            " ",
            question.lower().strip(),
        )

        # --------------------------------------------------------
        # Explicit conflict wording
        # --------------------------------------------------------

        explicit_conflict_phrases = (
            "documents disagree",
            "policies disagree",
            "policy conflict",
            "conflicting policy",
            "conflicting policies",
            "takes precedence",
            "take precedence",
            "which version applies",
            "which policy applies",
            "which rule applies",
            "which one applies",
            "which one is valid",
            "superseded",
            "supersedes",
        )

        if any(
            phrase in q
            for phrase
            in explicit_conflict_phrases
        ):
            return True

        # --------------------------------------------------------
        # Explicit old/new grace-rule wording
        # --------------------------------------------------------

        if re.search(
            r"\b(?:five|seven|5|7)[-\s]?day\b",
            q,
        ):
            return True

        # --------------------------------------------------------
        # Version/modifier + policy concept
        #
        # Handles:
        # latest collections rule
        # current collections policy
        # previous grace rule
        # old policy version
        # newer collections version
        #
        # Does NOT match:
        # mark account fully current
        # --------------------------------------------------------

        version_modifier = (
            r"\b(?:"
            r"latest|current|previous|"
            r"old|older|new|newer|active"
            r")\b"
        )

        policy_concept = (
            r"\b(?:"
            r"policy|policies|rule|rules|"
            r"version|versions|collections|"
            r"grace|grace[-\s]?period"
            r")\b"
        )

        # Modifier appears before policy concept.
        if re.search(
            rf"{version_modifier}"
            rf".{{0,40}}"
            rf"{policy_concept}",
            q,
        ):
            return True

        # Policy concept appears before modifier.
        if re.search(
            rf"{policy_concept}"
            rf".{{0,40}}"
            rf"{version_modifier}",
            q,
        ):
            return True

        # --------------------------------------------------------
        # "which" + policy/rule/version wording
        # --------------------------------------------------------

        if re.search(
            r"\bwhich\b"
            r".{0,40}"
            r"\b(?:policy|rule|version|collections)\b",
            q,
        ):
            return True

        return False

    # ============================================================
    # ANSWER
    # ============================================================

    async def answer(
        self,
        question: str,
        context=None,
    ):

        started = time.perf_counter()

        ctx = RAGContext.model_validate(
            context or {}
        )

        trace_rec = TraceRecorder()

        correction_actions = []
        retrieval_trace = []
        query_history = [
            question
        ]

        accumulated_chunks = []

        retrieval_ms = 0.0
        evaluation_ms = 0.0
        generation_ms = 0.0

        trace_rec.add(
            PipelineState.START,
            PipelineState.ANALYZE_QUERY,
            "Request received",
        )

        # --------------------------------------------------------
        # ANALYSIS
        # --------------------------------------------------------

        a0 = time.perf_counter()

        analysis = self.analyzer.analyze(
            question,
            ctx,
        )

        analysis_ms = (
            time.perf_counter()
            - a0
        ) * 1000

        # ========================================================
        # AMBIGUOUS
        # ========================================================

        if analysis.needs_clarification:

            trace_rec.add(
                PipelineState.ANALYZE_QUERY,
                PipelineState.ASK_CLARIFICATION,
                "Query is ambiguous",
            )

            trace_rec.add(
                PipelineState.ASK_CLARIFICATION,
                PipelineState.COMPLETE,
                "Clarification requested",
            )

            total = (
                time.perf_counter()
                - started
            ) * 1000

            return RAGResult(
                answer=None,
                answerable=False,
                confidence=analysis.confidence,
                verdict=EvidenceVerdict.AMBIGUOUS,
                retrieval_iterations=0,
                query_history=query_history,
                correction_actions=[
                    CorrectionAction.ASK_CLARIFICATION
                ],
                clarification_question=(
                    "Could you clarify the issue—for example "
                    "whether this is about a payment, hardship, "
                    "missing transaction, privacy, verification, "
                    "or human support?"
                ),
                state_transitions=(
                    trace_rec.transitions
                ),
                latency=LatencyBreakdown(
                    analysis_ms=analysis_ms,
                    total_ms=total,
                ),
            )

        # ========================================================
        # OUT OF SCOPE
        # ========================================================

        if (
            analysis.query_type.value
            == "out_of_scope"
        ):

            trace_rec.add(
                PipelineState.ANALYZE_QUERY,
                PipelineState.ABSTAIN,
                "Out-of-scope query",
            )

            trace_rec.add(
                PipelineState.ABSTAIN,
                PipelineState.COMPLETE,
                "Abstained safely",
            )

            total = (
                time.perf_counter()
                - started
            ) * 1000

            return RAGResult(
                answer=None,
                answerable=False,
                confidence=analysis.confidence,
                verdict=EvidenceVerdict.OUT_OF_SCOPE,
                retrieval_iterations=0,
                query_history=query_history,
                correction_actions=[
                    CorrectionAction.ABSTAIN
                ],
                abstention_reason=(
                    "The question is outside the supported "
                    "BFSI/customer-support knowledge base."
                ),
                state_transitions=(
                    trace_rec.transitions
                ),
                latency=LatencyBreakdown(
                    analysis_ms=analysis_ms,
                    total_ms=total,
                ),
            )

        current_query = question

        preferred_doc = None
        final_eval = None
        conflict_analysis = None

        final_generation_chunks = []

        # ========================================================
        # SELF-CORRECTION LOOP
        # ========================================================

        for iteration in range(
            1,
            self.config.max_retrieval_iterations
            + 1,
        ):

            from_state = (
                PipelineState.ANALYZE_QUERY
                if iteration == 1
                else PipelineState.REWRITE_QUERY
            )

            trace_rec.add(
                from_state,
                PipelineState.RETRIEVE,
                (
                    f"Retrieval iteration "
                    f"{iteration}"
                ),
            )

            # ----------------------------------------------------
            # RETRIEVE
            # ----------------------------------------------------

            r0 = time.perf_counter()

            if iteration == 1:

                (
                    iteration_chunks,
                    initial_action,
                ) = self._retrieve_for_analysis(
                    current_query,
                    analysis,
                )

                if (
                    initial_action
                    != CorrectionAction.NONE
                ):
                    correction_actions.append(
                        initial_action
                    )

            else:

                iteration_chunks = (
                    self.retriever.retrieve(
                        self._search_request(
                            current_query,
                            analysis,
                        )
                    )
                )

            retrieval_ms += (
                time.perf_counter()
                - r0
            ) * 1000

            accumulated_chunks = (
                self._merge_chunks(
                    accumulated_chunks,
                    iteration_chunks,
                )
            )

            trace_rec.add(
                PipelineState.RETRIEVE,
                PipelineState.EVALUATE_EVIDENCE,
                "Evaluate accumulated evidence",
            )

            # ----------------------------------------------------
            # EVALUATE
            # ----------------------------------------------------

            e0 = time.perf_counter()

            evaluation = (
                self.evaluator.evaluate(
                    question,
                    analysis,
                    accumulated_chunks,
                )
            )

            evaluation_ms += (
                time.perf_counter()
                - e0
            ) * 1000

            final_eval = evaluation

            relevant_chunks = (
                self._relevant_chunks(
                    accumulated_chunks,
                    evaluation.supported_document_ids,
                )
            )

            final_generation_chunks = (
                relevant_chunks
            )

            action = CorrectionAction.NONE

            # ====================================================
            # POLICY CONFLICT RESOLUTION
            # ====================================================

            if (
                evaluation.verdict
                == EvidenceVerdict.SUFFICIENT
                and self
                ._query_can_contain_policy_conflict(
                    question,
                    analysis,
                )
            ):

                conflict_analysis = (
                    self.conflicts.detect(
                        relevant_chunks
                    )
                )

                if (
                    conflict_analysis
                    .conflict_detected
                ):

                    correction_actions.append(
                        CorrectionAction
                        .CONFLICT_RESOLUTION
                    )

                    if (
                        conflict_analysis
                        .resolution_status
                        == ConflictResolutionStatus
                        .RESOLVED
                    ):

                        preferred_doc = (
                            conflict_analysis
                            .preferred_document_id
                        )

                    else:

                        evaluation.verdict = (
                            EvidenceVerdict.CONFLICT
                        )

            # ====================================================
            # SUCCESS
            # ====================================================

            if (
                evaluation.verdict
                == EvidenceVerdict.SUFFICIENT
            ):

                retrieval_trace.append(
                    RetrievalTrace(
                        iteration=iteration,
                        query=current_query,
                        retrieved_chunk_ids=[
                            chunk.chunk_id
                            for chunk
                            in accumulated_chunks
                        ],
                        verdict=(
                            evaluation.verdict
                        ),
                        action=(
                            CorrectionAction.NONE
                        ),
                        reason=(
                            evaluation.reason
                        ),
                    )
                )

                break

            # ====================================================
            # CONFLICT
            # ====================================================

            if (
                evaluation.verdict
                == EvidenceVerdict.CONFLICT
            ):

                action = (
                    CorrectionAction
                    .CONFLICT_RESOLUTION
                )

                conflict_analysis = (
                    self.conflicts.detect(
                        relevant_chunks
                    )
                )

                retrieval_trace.append(
                    RetrievalTrace(
                        iteration=iteration,
                        query=current_query,
                        retrieved_chunk_ids=[
                            chunk.chunk_id
                            for chunk
                            in accumulated_chunks
                        ],
                        verdict=(
                            evaluation.verdict
                        ),
                        action=action,
                        reason=(
                            evaluation.reason
                        ),
                    )
                )

                if (
                    conflict_analysis
                    .resolution_status
                    == ConflictResolutionStatus
                    .RESOLVED
                ):

                    preferred_doc = (
                        conflict_analysis
                        .preferred_document_id
                    )

                    final_eval.verdict = (
                        EvidenceVerdict.SUFFICIENT
                    )

                    correction_actions.append(
                        action
                    )

                    final_generation_chunks = (
                        relevant_chunks
                    )

                    break

                correction_actions.extend(
                    [
                        action,
                        CorrectionAction.ABSTAIN,
                    ]
                )

                break

            # ====================================================
            # MAX ITERATIONS
            # ====================================================

            if (
                iteration
                >= self.config
                .max_retrieval_iterations
            ):

                action = (
                    CorrectionAction.ABSTAIN
                )

                correction_actions.append(
                    action
                )

                retrieval_trace.append(
                    RetrievalTrace(
                        iteration=iteration,
                        query=current_query,
                        retrieved_chunk_ids=[
                            chunk.chunk_id
                            for chunk
                            in accumulated_chunks
                        ],
                        verdict=(
                            evaluation.verdict
                        ),
                        action=action,
                        reason=(
                            "Maximum retrieval "
                            "iterations reached."
                        ),
                    )
                )

                break

            # ====================================================
            # SELF-CORRECTION
            # ====================================================

            if (
                evaluation.verdict
                == EvidenceVerdict.LOW_RELEVANCE
            ):

                action = (
                    CorrectionAction.QUERY_REWRITE
                )

            else:

                action = (
                    CorrectionAction.BROADEN_RETRIEVAL
                )

            current_query = (
                self.rewriter.rewrite(
                    question,
                    analysis,
                    iteration,
                )
            )

            correction_actions.append(
                action
            )

            query_history.append(
                current_query
            )

            retrieval_trace.append(
                RetrievalTrace(
                    iteration=iteration,
                    query=query_history[-2],
                    retrieved_chunk_ids=[
                        chunk.chunk_id
                        for chunk
                        in accumulated_chunks
                    ],
                    verdict=(
                        evaluation.verdict
                    ),
                    action=action,
                    reason=(
                        evaluation.reason
                    ),
                )
            )

            trace_rec.add(
                PipelineState.EVALUATE_EVIDENCE,
                PipelineState.REWRITE_QUERY,
                action.value,
            )

        # ========================================================
        # GENERATE ANSWER
        # ========================================================

        if (
            final_eval
            and final_eval.verdict
            == EvidenceVerdict.SUFFICIENT
        ):

            trace_rec.add(
                PipelineState.EVALUATE_EVIDENCE,
                PipelineState.GENERATE_ANSWER,
                "Evidence is sufficient",
            )

            g0 = time.perf_counter()

            generation_chunks = (
                final_generation_chunks
                or accumulated_chunks
            )

            generation_chunks = (
                self._prefer_active_policy_versions(
                    question,
                    generation_chunks,
                )
            )

            (
                answer,
                citations,
            ) = self.generator.generate(
                question,
                generation_chunks,
                preferred_doc,
            )

            generation_ms = (
                time.perf_counter()
                - g0
            ) * 1000

            trace_rec.add(
                PipelineState.GENERATE_ANSWER,
                PipelineState.COMPLETE,
                "Grounded answer generated",
            )

            total = (
                time.perf_counter()
                - started
            ) * 1000

            return RAGResult(
                answer=answer,
                answerable=True,
                confidence=(
                    final_eval.confidence
                ),
                verdict=(
                    EvidenceVerdict.SUFFICIENT
                ),
                retrieval_iterations=(
                    len(retrieval_trace)
                    or 1
                ),
                query_history=query_history,
                retrieved_chunks=(
                    accumulated_chunks
                ),
                conflict_detected=bool(
                    conflict_analysis
                    and conflict_analysis
                    .conflict_detected
                ),
                conflict_resolution=(
                    conflict_analysis
                ),
                correction_actions=(
                    correction_actions
                ),
                citations=citations,
                trace=retrieval_trace,
                state_transitions=(
                    trace_rec.transitions
                ),
                latency=LatencyBreakdown(
                    analysis_ms=analysis_ms,
                    retrieval_ms=retrieval_ms,
                    evaluation_ms=(
                        evaluation_ms
                    ),
                    generation_ms=(
                        generation_ms
                    ),
                    total_ms=total,
                ),
            )

        # ========================================================
        # SAFE ABSTENTION
        # ========================================================

        verdict = (
            final_eval.verdict
            if final_eval
            else EvidenceVerdict.INSUFFICIENT
        )

        if (
            verdict
            == EvidenceVerdict.LOW_RELEVANCE
        ):
            verdict = (
                EvidenceVerdict.INSUFFICIENT
            )

        trace_rec.add(
            PipelineState.EVALUATE_EVIDENCE,
            PipelineState.ABSTAIN,
            "Evidence remained unsafe or insufficient",
        )

        trace_rec.add(
            PipelineState.ABSTAIN,
            PipelineState.COMPLETE,
            "Abstained safely",
        )

        total = (
            time.perf_counter()
            - started
        ) * 1000

        return RAGResult(
            answer=None,
            answerable=False,
            confidence=(
                final_eval.confidence
                if final_eval
                else 0.8
            ),
            verdict=verdict,
            retrieval_iterations=(
                len(retrieval_trace)
            ),
            query_history=query_history,
            retrieved_chunks=(
                accumulated_chunks
            ),
            conflict_detected=bool(
                conflict_analysis
                and conflict_analysis
                .conflict_detected
            ),
            conflict_resolution=(
                conflict_analysis
            ),
            correction_actions=(
                correction_actions
                or [
                    CorrectionAction.ABSTAIN
                ]
            ),
            abstention_reason=(
                final_eval.reason
                if final_eval
                else (
                    "Insufficient policy evidence."
                )
            ),
            trace=retrieval_trace,
            state_transitions=(
                trace_rec.transitions
            ),
            latency=LatencyBreakdown(
                analysis_ms=analysis_ms,
                retrieval_ms=retrieval_ms,
                evaluation_ms=evaluation_ms,
                generation_ms=generation_ms,
                total_ms=total,
            ),
        )