from __future__ import annotations

import re

from .schemas import Citation


class GroundedAnswerGenerator:
    """
    Deterministic extractive grounded-answer generator.

    It selects question-relevant policy statements rather than
    blindly returning the beginning of retrieved chunks.
    """

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
        "by",
        "with",
        "and",
        "or",
        "but",
        "if",
        "then",
        "that",
        "this",
        "it",
        "as",
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
            "hardship",
            "financial",
            "difficulty",
            "relief",
            "job",
            "income",
            "unemployment",
            "unemployed",
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
            "credential",
            "credentials",
            "security",
        },
        {
            "human",
            "person",
            "agent",
            "specialist",
            "escalate",
            "escalation",
            "callback",
            "call",
        },
        {
            "privacy",
            "third",
            "party",
            "share",
            "reveal",
            "disclose",
            "disclosure",
        },
        {
            "missing",
            "posting",
            "posted",
            "update",
            "updated",
            "reflected",
            "dispute",
            "transaction",
        },
        {
            "grace",
            "buffer",
            "days",
            "overdue",
            "due",
            "collections",
        },
        {
            "promise",
            "guarantee",
            "guaranteed",
            "approval",
            "approved",
        },
    ]

    def generate(
        self,
        question: str,
        chunks,
        preferred_document_id: str | None = None,
    ):

        evidence = list(
            chunks
        )

        if not evidence:
            return None, []

        # =========================================================
        # RESOLVED CONFLICT
        # =========================================================

        if preferred_document_id:

            preferred = [
                chunk
                for chunk in evidence
                if (
                    chunk.document_id
                    == preferred_document_id
                )
            ]

            if preferred:
                evidence = preferred

        # =========================================================
        # SUPPORT-MACRO NOISE CONTROL
        # =========================================================

        #
        # support_macros is useful for fallback language, but it
        # should not dominate normal policy answers.
        #

        question_lower = (
            question.lower()
        )

        needs_macro = any(
            phrase in question_lower
            for phrase in (
                "what should support say",
                "what should the agent say",
                "clarification",
                "insufficient evidence",
                "conflicting policy",
            )
        )

        if not needs_macro:

            without_macros = [
                chunk
                for chunk in evidence
                if (
                    chunk.document_id
                    != "support_macros"
                )
            ]

            if without_macros:
                evidence = (
                    without_macros
                )

        # =========================================================
        # CANDIDATE SENTENCES
        # =========================================================

        question_terms = (
            self._expanded_tokens(
                question
            )
        )

        candidates = []

        for chunk_rank, chunk in enumerate(
            evidence,
            start=1,
        ):

            sentences = (
                self._split_sentences(
                    chunk.text
                )
            )

            for sentence_index, sentence in enumerate(
                sentences
            ):

                score = (
                    self._score_sentence(
                        question=question,
                        question_terms=question_terms,
                        sentence=sentence,
                        chunk=chunk,
                        chunk_rank=chunk_rank,
                        sentence_index=sentence_index,
                    )
                )

                candidates.append(
                    {
                        "sentence":
                            sentence,

                        "score":
                            score,

                        "chunk":
                            chunk,
                    }
                )

        if not candidates:
            return None, []

        candidates.sort(
            key=lambda item: (
                item["score"]
            ),
            reverse=True,
        )

        # =========================================================
        # SELECT SENTENCES
        # =========================================================

        selected = []
        selected_text = set()
        selected_documents = set()

        # First pick the strongest sentence.
        for candidate in candidates:

            sentence = candidate[
                "sentence"
            ].strip()

            normalized = (
                self._normalize_sentence(
                    sentence
                )
            )

            if not normalized:
                continue

            selected.append(
                candidate
            )

            selected_text.add(
                normalized
            )

            selected_documents.add(
                candidate[
                    "chunk"
                ].document_id
            )

            break

        # ---------------------------------------------------------
        # MULTI-ASPECT QUESTIONS:
        # try to include evidence from another strong document.
        # ---------------------------------------------------------

        multi_aspect = any(
            phrase in question_lower
            for phrase in (
                " and ",
                " also ",
                "what policies",
                "what rules",
                "what options",
                "before changing",
                "third party",
            )
        )

        if multi_aspect:

            for candidate in candidates:

                document_id = (
                    candidate[
                        "chunk"
                    ].document_id
                )

                sentence = (
                    candidate[
                        "sentence"
                    ].strip()
                )

                normalized = (
                    self._normalize_sentence(
                        sentence
                    )
                )

                if (
                    not normalized
                    or normalized
                    in selected_text
                    or document_id
                    in selected_documents
                ):
                    continue

                # Do not add weak unrelated secondary evidence.
                if (
                    candidate["score"]
                    < candidates[0]["score"]
                    * 0.38
                ):
                    continue

                selected.append(
                    candidate
                )

                selected_text.add(
                    normalized
                )

                selected_documents.add(
                    document_id
                )

                break

        # Fill remaining answer slots by relevance.
        for candidate in candidates:

            if len(selected) >= 3:
                break

            sentence = (
                candidate[
                    "sentence"
                ].strip()
            )

            normalized = (
                self._normalize_sentence(
                    sentence
                )
            )

            if (
                not normalized
                or normalized
                in selected_text
            ):
                continue

            selected.append(
                candidate
            )

            selected_text.add(
                normalized
            )

            selected_documents.add(
                candidate[
                    "chunk"
                ].document_id
            )

        if not selected:
            return None, []

        answer = (
            "Based on the available policy: "
            + " ".join(
                item[
                    "sentence"
                ].strip()
                for item in selected
            )
        )

        # =========================================================
        # CITATIONS
        # =========================================================

        citations = []
        cited_chunk_ids = set()

        for item in selected:

            chunk = item[
                "chunk"
            ]

            if (
                chunk.chunk_id
                in cited_chunk_ids
            ):
                continue

            cited_chunk_ids.add(
                chunk.chunk_id
            )

            citations.append(
                Citation(
                    document_id=(
                        chunk.document_id
                    ),
                    chunk_id=(
                        chunk.chunk_id
                    ),
                    title=(
                        chunk.metadata.title
                    ),
                    version=(
                        chunk.metadata.version
                    ),
                    effective_date=(
                        chunk.metadata
                        .effective_date
                    ),
                )
            )

        return (
            answer,
            citations,
        )

    # =============================================================
    # SCORING
    # =============================================================

    def _score_sentence(
        self,
        question: str,
        question_terms: set[str],
        sentence: str,
        chunk,
        chunk_rank: int,
        sentence_index: int,
    ) -> float:

        sentence_terms = (
            self._expanded_tokens(
                sentence
            )
        )

        if question_terms:

            overlap = (
                len(
                    question_terms
                    & sentence_terms
                )
                / len(
                    question_terms
                )
            )

        else:
            overlap = 0.0

        score = (
            overlap
            * 6.0
        )

        direct_question = (
            self._tokens(
                question
            )
        )

        direct_sentence = (
            self._tokens(
                sentence
            )
        )

        direct_matches = len(
            direct_question
            & direct_sentence
        )

        score += (
            direct_matches
            * 0.70
        )

        lower_q = (
            question.lower()
        )

        lower_s = (
            sentence.lower()
        )

        # ---------------------------------------------------------
        # POLICY / SAFETY LANGUAGE
        # ---------------------------------------------------------

        policy_phrases = {
            "must not": 1.2,
            "must ": 0.45,
            "do not": 1.2,
            "cannot": 0.9,
            "may not": 0.9,
            "should not": 0.8,
            "not automatically": 1.0,
            "not guaranteed": 1.0,
            "must never": 1.2,
        }

        for phrase, bonus in (
            policy_phrases.items()
        ):

            if phrase in lower_s:
                score += bonus

        # ---------------------------------------------------------
        # HARDSHIP OPTIONS
        # ---------------------------------------------------------

        if any(
            phrase in lower_q
            for phrase in (
                "what options",
                "financial difficulty",
                "temporary relief",
                "hardship options",
            )
        ):

            if any(
                phrase in lower_s
                for phrase in (
                    "possible outcomes",
                    "reduced-payment",
                    "changed payment date",
                    "short deferral",
                    "hardship team",
                )
            ):
                score += 7.0

        # ---------------------------------------------------------
        # WRONG NUMBER / DISCLOSURE
        # ---------------------------------------------------------

        if any(
            phrase in lower_q
            for phrase in (
                "wrong number",
                "belongs to someone else",
                "account holder",
                "third party",
                "someone else",
                "how much i owe",
            )
        ):

            if (
                "do not disclose"
                in lower_s
            ):
                score += 7.0

            if (
                "must not reveal"
                in lower_s
            ):
                score += 6.0

        # ---------------------------------------------------------
        # OTP / PIN
        # ---------------------------------------------------------

        if (
            "otp" in lower_q
            or "pin" in lower_q
            or "one-time password"
            in lower_q
        ):

            if (
                "one-time password"
                in lower_s
            ):
                score += 8.0

            if (
                "pin"
                in lower_s
            ):
                score += 6.0

            if (
                "must never ask"
                in lower_s
            ):
                score += 4.0

        # ---------------------------------------------------------
        # PARTIAL PAYMENT
        # ---------------------------------------------------------

        partial_query = any(
            phrase in lower_q
            for phrase in (
                "partial payment",
                "paying part",
                "pay only part",
                "paid a little",
                "half payment",
                "part of the amount",
                "instalment",
                "installment",
            )
        )

        if partial_query:

            if (
                "partial payment"
                in lower_s
            ):
                score += 7.0

            if (
                "does not automatically"
                in lower_s
            ):
                score += 7.0

            if (
                "overdue status"
                in lower_s
            ):
                score += 4.0

        # ---------------------------------------------------------
        # HUMAN / CALLBACK
        # ---------------------------------------------------------

        if any(
            phrase in lower_q
            for phrase in (
                "human",
                "person handle",
                "speak to someone",
                "call at exactly",
                "callback",
                "call me back",
            )
        ):

            if (
                "human specialist"
                in lower_s
            ):
                score += 6.0

            if (
                "do not promise an exact callback time"
                in lower_s
            ):
                score += 8.0

        # ---------------------------------------------------------
        # PAYMENT DISPUTE
        # ---------------------------------------------------------

        if any(
            phrase in lower_q
            for phrase in (
                "payment succeeded",
                "payment cleared",
                "cannot see it",
                "missing payment",
                "not showing",
                "update nahi",
            )
        ):

            if (
                "payment disputes"
                in lower_s
            ):
                score += 4.0

            if (
                "transaction evidence"
                in lower_s
            ):
                score += 5.0

            if (
                "must not claim"
                in lower_s
            ):
                score += 5.0

        # ---------------------------------------------------------
        # GRACE PERIOD / VERSION
        # ---------------------------------------------------------

        if any(
            phrase in lower_q
            for phrase in (
                "grace",
                "buffer",
                "extra time",
                "five-day",
                "seven-day",
                "5 day",
                "7 day",
            )
        ):

            if (
                "grace period is 7 days"
                in lower_s
            ):
                score += 10.0

            elif (
                "7 days"
                in lower_s
            ):
                score += 7.0

            if (
                "supersedes"
                in lower_s
            ):
                score += 4.0

        # ---------------------------------------------------------
        # PROMISE / APPROVAL
        # ---------------------------------------------------------

        if any(
            phrase in lower_q
            for phrase in (
                "approved",
                "approval",
                "guarantee",
                "definitely approved",
            )
        ):

            if (
                "must not promise approval"
                in lower_s
            ):
                score += 9.0

            if (
                "not guaranteed"
                in lower_s
            ):
                score += 7.0

        # ---------------------------------------------------------
        # RETRIEVAL SCORE
        # ---------------------------------------------------------

        semantic = max(
            0.0,
            min(
                1.0,
                float(
                    chunk.score
                ),
            ),
        )

        score += (
            semantic
            * 1.5
        )

        score += (
            0.30
            / max(
                1,
                chunk_rank,
            )
        )

        # Very small sentence-order preference only.
        score += (
            0.03
            / (
                sentence_index
                + 1
            )
        )

        return score

    # =============================================================
    # TOKENIZATION
    # =============================================================

    def _tokens(
        self,
        text: str,
    ) -> set[str]:

        return {
            token
            for token in re.findall(
                r"[a-z0-9]+",
                text.lower(),
            )
            if (
                len(token) > 2
                and token
                not in self.STOP_WORDS
            )
        }

    def _expanded_tokens(
        self,
        text: str,
    ) -> set[str]:

        tokens = (
            self._tokens(
                text
            )
        )

        expanded = set(
            tokens
        )

        for group in (
            self.SYNONYM_GROUPS
        ):

            if tokens & group:
                expanded.update(
                    group
                )

        return expanded

    # =============================================================
    # SENTENCE SPLITTING
    # =============================================================

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:

        cleaned = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if not cleaned:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+",
            cleaned,
        )

        output = []

        for sentence in sentences:

            sentence = (
                sentence.strip()
            )

            sentence = re.sub(
                r"^#+\s*",
                "",
                sentence,
            )

            if sentence:
                output.append(
                    sentence
                )

        return output

    @staticmethod
    def _normalize_sentence(
        sentence: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            sentence.lower(),
        ).strip()