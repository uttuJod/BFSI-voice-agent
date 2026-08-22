from __future__ import annotations

import re

from .conflict_detector import ConflictDetector
from .schemas import (
    EvidenceEvaluation,
    EvidenceVerdict,
)


STOP_WORDS = {
    "a",
    "an",
    "the",
    "i",
    "me",
    "my",
    "you",
    "your",
    "we",
    "our",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "and",
    "or",
    "but",
    "if",
    "then",
    "this",
    "that",
    "it",
    "with",
    "from",
    "what",
    "when",
    "where",
    "why",
    "how",
    "can",
    "could",
    "should",
    "would",
    "will",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "please",
    "tell",
}


SYNONYM_GROUPS = [
    {
        "pay",
        "paid",
        "payment",
        "repayment",
        "amount",
        "owe",
        "owing",
        "balance",
        "debt",
    },
    {
        "partial",
        "part",
        "half",
        "little",
        "instalment",
        "instalments",
        "installment",
        "installments",
    },
    {
        "job",
        "employment",
        "unemployed",
        "unemployment",
        "income",
        "hardship",
        "difficulty",
        "relief",
        "financial",
    },
    {
        "verify",
        "verified",
        "verification",
        "identity",
        "authentication",
    },
    {
        "otp",
        "password",
        "pin",
        "security",
        "credential",
        "credentials",
    },
    {
        "human",
        "person",
        "agent",
        "specialist",
        "escalation",
        "escalate",
        "callback",
        "call",
    },
    {
        "privacy",
        "private",
        "third",
        "party",
        "disclose",
        "disclosure",
        "share",
        "reveal",
    },
    {
        "missing",
        "posted",
        "posting",
        "showing",
        "shown",
        "reflected",
        "dispute",
        "transaction",
        "updated",
    },
    {
        "grace",
        "buffer",
        "extra",
        "days",
        "overdue",
        "collections",
        "due",
    },
    {
        "promise",
        "guarantee",
        "guaranteed",
        "approval",
        "approved",
    },
]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
        if (
            len(token) > 2
            and token not in STOP_WORDS
        )
    }


def _expand(tokens: set[str]) -> set[str]:
    expanded = set(tokens)

    for group in SYNONYM_GROUPS:
        if tokens & group:
            expanded.update(group)

    return expanded


def _lexical_overlap(
    question: str,
    text: str,
) -> float:

    q = _expand(
        _tokens(question)
    )

    if not q:
        return 0.0

    evidence = _expand(
        _tokens(text)
    )

    return (
        len(q & evidence)
        / len(q)
    )


def _aggregate_overlap(
    question: str,
    chunks,
) -> float:

    if not chunks:
        return 0.0

    combined = " ".join(
        chunk.text
        for chunk in chunks[:5]
    )

    return _lexical_overlap(
        question,
        combined,
    )


class EvidenceEvaluator:
    """
    Deterministic evidence evaluation.

    Important design rule:
    query-analyzer domain is a SOFT signal.

    Strong retrieved evidence is allowed to override imperfect
    lexical matching, which is particularly important for
    paraphrased and Hinglish questions.
    """

    def __init__(
        self,
        low_relevance_threshold: float = 0.22,
        sufficient_threshold: float = 0.32,
        strong_relevance_threshold: float = 0.42,
    ):
        self.low = low_relevance_threshold
        self.sufficient = sufficient_threshold
        self.strong = strong_relevance_threshold

        self.conflicts = ConflictDetector()

    # ============================================================
    # SUPPORT RANKING
    # ============================================================

    def _rank_supported_chunks(
        self,
        question,
        analysis,
        chunks,
    ):

        ranked = []

        for chunk in chunks:

            semantic = float(
                chunk.score
            )

            lexical = _lexical_overlap(
                question,
                chunk.text,
            )

            domain_match = (
                bool(analysis.domain)
                and chunk.metadata.domain
                == analysis.domain
            )

            combined = (
                semantic
                + (0.28 * lexical)
                + (
                    0.07
                    if domain_match
                    else 0.0
                )
            )

            ranked.append(
                (
                    combined,
                    semantic,
                    lexical,
                    domain_match,
                    chunk,
                )
            )

        ranked.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return ranked

    def _supported_chunks(
        self,
        question,
        analysis,
        chunks,
    ):

        ranked = self._rank_supported_chunks(
            question,
            analysis,
            chunks,
        )

        if not ranked:
            return []

        best_combined = ranked[0][0]
        best_semantic = max(
            row[1]
            for row in ranked
        )

        supported = []

        for (
            combined,
            semantic,
            lexical,
            domain_match,
            chunk,
        ) in ranked:

            keep = False

            # Near the best combined result.
            if (
                combined
                >= best_combined - 0.15
            ):
                keep = True

            # Good direct lexical support.
            if lexical >= 0.16:
                keep = True

            # Strong semantic support.
            if semantic >= max(
                self.sufficient,
                best_semantic - 0.08,
            ):
                keep = True

            # Domain match should not be thrown away merely because
            # wording is paraphrased.
            if (
                domain_match
                and semantic
                >= self.low * 0.75
            ):
                keep = True

            if keep:
                supported.append(
                    chunk
                )

        return supported[:5]

    # ============================================================
    # MAIN
    # ============================================================

    def evaluate(
        self,
        question,
        analysis,
        chunks,
    ):

        # --------------------------------------------------------
        # AMBIGUOUS
        # --------------------------------------------------------

        if analysis.needs_clarification:

            return EvidenceEvaluation(
                verdict=(
                    EvidenceVerdict.AMBIGUOUS
                ),
                confidence=0.98,
                relevance_score=0.0,
                reason=(
                    analysis.clarification_reason
                    or "The request requires clarification."
                ),
            )

        # --------------------------------------------------------
        # OUT OF SCOPE
        # --------------------------------------------------------

        if (
            analysis.query_type.value
            == "out_of_scope"
        ):

            return EvidenceEvaluation(
                verdict=(
                    EvidenceVerdict.OUT_OF_SCOPE
                ),
                confidence=0.99,
                relevance_score=0.0,
                reason=(
                    "The request is outside the "
                    "supported BFSI knowledge base."
                ),
            )

        # --------------------------------------------------------
        # KNOWN IN-DOMAIN BUT UNSUPPORTED ACTION
        # --------------------------------------------------------

        if (
            analysis.domain
            == "account_management"
        ):

            return EvidenceEvaluation(
                verdict=(
                    EvidenceVerdict.INSUFFICIENT
                ),
                confidence=0.98,
                relevance_score=0.0,
                missing_information=[
                    (
                        "The knowledge base does not contain "
                        "a policy for the requested "
                        "account-management action."
                    )
                ],
                reason=(
                    "The request is BFSI-related but "
                    "unsupported by the available corpus."
                ),
            )

        # --------------------------------------------------------
        # NO RETRIEVAL
        # --------------------------------------------------------

        if not chunks:

            return EvidenceEvaluation(
                verdict=(
                    EvidenceVerdict.INSUFFICIENT
                ),
                confidence=0.96,
                relevance_score=0.0,
                missing_information=[
                    "No relevant policy evidence was retrieved."
                ],
                reason=(
                    "No qualifying evidence was retrieved."
                ),
            )

        supported = self._supported_chunks(
            question,
            analysis,
            chunks,
        )

        if not supported:

            return EvidenceEvaluation(
                verdict=(
                    EvidenceVerdict.LOW_RELEVANCE
                ),
                confidence=0.88,
                relevance_score=0.0,
                reason=(
                    "Retrieved evidence does not "
                    "directly support the request."
                ),
            )

        semantic_scores = [
            float(chunk.score)
            for chunk in supported
        ]

        best_semantic = max(
            semantic_scores
        )

        top_scores = (
            semantic_scores[:3]
        )

        avg_semantic = (
            sum(top_scores)
            / len(top_scores)
        )

        lexical = _aggregate_overlap(
            question,
            supported,
        )

        domain_chunks = [
            chunk
            for chunk in supported
            if (
                analysis.domain
                and chunk.metadata.domain
                == analysis.domain
            )
        ]

        domain_match = bool(
            domain_chunks
        )

        best_domain_score = (
            max(
                float(chunk.score)
                for chunk in domain_chunks
            )
            if domain_chunks
            else 0.0
        )

        # --------------------------------------------------------
        # CONFLICT
        # --------------------------------------------------------

        conflict = (
            self.conflicts.detect(
                supported
            )
        )

        if (
            conflict.conflict_detected
            and conflict.resolution_status.value
            == "unresolved"
        ):

            return EvidenceEvaluation(
                verdict=(
                    EvidenceVerdict.CONFLICT
                ),
                confidence=0.95,
                relevance_score=max(
                    0.0,
                    min(
                        1.0,
                        best_semantic,
                    ),
                ),
                conflict_detected=True,
                supported_document_ids=list(
                    dict.fromkeys(
                        chunk.document_id
                        for chunk in supported
                    )
                ),
                reason=(
                    "Relevant policy evidence contains "
                    "an unresolved conflict."
                ),
            )

        # --------------------------------------------------------
        # SUFFICIENT RULES
        # --------------------------------------------------------

        sufficient = False
        reason = ""

        # Strong semantic retrieval is enough.
        if (
            best_semantic
            >= self.strong
        ):

            sufficient = True

            reason = (
                "Strong semantic evidence supports "
                "a grounded answer."
            )

        # Strong domain evidence should survive paraphrasing.
        elif (
            domain_match
            and best_domain_score
            >= self.low * 0.75
        ):

            sufficient = True

            reason = (
                "Retrieved evidence matches the predicted "
                "policy domain with adequate semantic support."
            )

        # Semantic + lexical combination.
        elif (
            best_semantic
            >= self.low
            and lexical >= 0.10
        ):

            sufficient = True

            reason = (
                "Semantic and lexical evidence jointly "
                "support a grounded answer."
            )

        # A good collection of moderately relevant evidence.
        elif (
            avg_semantic
            >= self.low
            and lexical >= 0.08
        ):

            sufficient = True

            reason = (
                "Multiple related evidence chunks jointly "
                "support the answer."
            )

        if sufficient:

            relevance = max(
                0.0,
                min(
                    1.0,
                    (
                        0.70
                        * best_semantic
                    )
                    + (
                        0.15
                        * avg_semantic
                    )
                    + (
                        0.15
                        * lexical
                    ),
                ),
            )

            return EvidenceEvaluation(
                verdict=(
                    EvidenceVerdict.SUFFICIENT
                ),
                confidence=min(
                    0.96,
                    (
                        0.72
                        + 0.20
                        * max(
                            best_semantic,
                            best_domain_score,
                        )
                        + 0.08
                        * lexical
                    ),
                ),
                relevance_score=relevance,
                conflict_detected=(
                    conflict.conflict_detected
                ),
                supported_document_ids=list(
                    dict.fromkeys(
                        chunk.document_id
                        for chunk in supported
                    )
                ),
                reason=reason,
            )

        # --------------------------------------------------------
        # LOW RELEVANCE VS INSUFFICIENT
        # --------------------------------------------------------

        if (
            best_semantic
            < self.low * 0.75
            and lexical < 0.08
        ):

            verdict = (
                EvidenceVerdict.LOW_RELEVANCE
            )

            reason = (
                "Retrieved evidence is weakly related "
                "to the request."
            )

        else:

            verdict = (
                EvidenceVerdict.INSUFFICIENT
            )

            reason = (
                "Evidence is related but not strong enough "
                "to safely answer."
            )

        return EvidenceEvaluation(
            verdict=verdict,
            confidence=0.84,
            relevance_score=max(
                0.0,
                min(
                    1.0,
                    (
                        0.75
                        * best_semantic
                    )
                    + (
                        0.25
                        * lexical
                    ),
                ),
            ),
            conflict_detected=(
                conflict.conflict_detected
            ),
            supported_document_ids=list(
                dict.fromkeys(
                    chunk.document_id
                    for chunk in supported
                )
            ),
            reason=reason,
        )