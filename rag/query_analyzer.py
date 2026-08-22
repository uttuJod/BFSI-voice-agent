from __future__ import annotations

import re
from collections import defaultdict

from .schemas import QueryAnalysis, QueryType


DOMAIN_TERMS: dict[str, dict[str, float]] = {
    "hardship": {
        "hardship": 3.0,
        "lost my job": 4.0,
        "job loss": 4.0,
        "unemployed": 4.0,
        "unemployment": 4.0,
        "reduced income": 3.5,
        "income disappeared": 4.0,
        "financial difficulty": 3.5,
        "financial shock": 3.5,
        "cannot pay": 3.0,
        "can't pay": 3.0,
        "unable to pay": 3.0,
        "cannot meet": 2.5,
        "can't meet": 2.5,
        "job chali gayi": 4.0,
        "emi bharna mushkil": 4.0,
        "pay nahi ho rahi": 3.5,
        "payment nahi ho rahi": 3.5,
        "temporary relief": 2.5,
    },

    "collections": {
        "collections": 3.0,
        "collection": 2.5,
        "grace period": 4.0,
        "grace": 3.0,
        "buffer after": 3.0,
        "extra time": 2.5,
        "overdue": 3.0,
        "late payment": 3.0,
        "promise to pay": 4.0,
        "promise-to-pay": 4.0,
        "due date": 2.0,
        "latest rule": 2.5,
        "current rule": 2.5,
        "five-day": 4.0,
        "seven-day": 4.0,
        "superseded": 4.0,
    },

    "repayment": {
        "repayment": 3.0,
        "partial payment": 4.5,
        "part payment": 4.0,
        "pay only part": 4.0,
        "paid a little": 4.0,
        "half payment": 4.0,
        "split payment": 4.0,
        "pay in parts": 4.0,
        "instalment": 3.5,
        "installment": 3.5,
        "smaller scheduled amounts": 3.0,
        "structured repayment": 4.0,
        "thoda amount": 4.0,
        "baaki baad": 4.0,
    },

    "dispute": {
        "payment dispute": 4.0,
        "payment cleared": 4.0,
        "payment succeeded": 4.0,
        "bank says": 2.5,
        "not showing": 4.0,
        "cannot see it": 3.5,
        "can't see it": 3.5,
        "not appeared": 3.5,
        "hasn't appeared": 3.5,
        "not reflected": 3.5,
        "not updated": 3.5,
        "system disagrees": 4.0,
        "records disagree": 4.0,
        "show nahi": 4.0,
        "update nahi hua": 4.0,
        "payment kar diya": 2.5,
        "transaction evidence": 3.0,
        "missing payment": 4.0,
    },

    "privacy": {
        "privacy": 4.0,
        "third party": 4.0,
        "someone else": 3.5,
        "another person": 3.5,
        "partner": 3.0,
        "wife": 3.0,
        "husband": 3.0,
        "stranger": 3.0,
        "account balance": 3.5,
        "payment history": 3.5,
        "share": 1.5,
        "disclose": 3.0,
        "tell them": 2.0,
    },

    "identity": {
        "identity": 4.0,
        "verification": 4.0,
        "verify": 3.5,
        "otp": 4.5,
        "one-time password": 4.5,
        "one time password": 4.5,
        "pin": 3.5,
        "authentication": 3.5,
        "proof step": 3.0,
    },

    "escalation": {
        "human": 4.0,
        "person handle": 4.0,
        "speak to someone": 4.0,
        "speak to a person": 4.0,
        "specialist": 3.0,
        "escalate": 3.5,
        "escalation": 3.5,
        "callback": 3.5,
        "call back": 3.5,
    },

    "support": {
        "support hours": 4.0,
        "opening hours": 4.0,
        "support open": 4.0,
        "support close": 4.0,
        "saturday": 2.5,
        "already paid": 3.0,
        "pay again": 3.0,
        "already made the payment": 3.0,
    },
}


UNSUPPORTED_ACCOUNT_PATTERNS = [
    r"\b(close|cancel|delete|terminate)\b.*\baccount\b",
    r"\b(change|update)\b.*\b(email|e-mail|mobile number|phone number|address)\b",
    r"\b(new|replace|replacement)\b.*\b(debit|credit)?\s*card\b",
    r"\b(credit limit|increase.*limit)\b",
    r"\b(tax statement|annual statement)\b",
    r"\b(add|change)\b.*\bnominee\b",
    r"\b(joint account|convert.*joint)\b",
    r"\b(reset|change|forgot)\b.*\b(mobile banking|online banking|banking).*"
    r"\bpassword\b",
]


OUT_OF_SCOPE_TERMS = {
    "weather",
    "forecast",
    "football",
    "soccer",
    "cricket",
    "world cup",
    "recipe",
    "cook",
    "pasta",
    "movie",
    "film",
    "photosynthesis",
    "planet",
    "mars",
    "mountain",
    "geography",
    "capital of",
    "invented",
    "inventor",
    "president of",
    "prime minister of",
    "bitcoin trading",
    "crypto price",
    "stock price",
    "exchange rate today",
}


GENERIC_VAGUE_PATTERNS = [
    r"^what (should|can) i do( now)?$",
    r"^what are my options$",
    r"^what happens next$",
    r"^can (you )?(help|assist|sort|fix).*$",
    r"^please (help|assist|tell me what to do).*$",
    r"^i need (help|assistance|support)$",
    r"^something is wrong$",
    r"^i have (a )?(problem|issue)( with my account)?$",
    r"^what now$",
]


HINGLISH_MARKERS = {
    "kya",
    "hai",
    "nahi",
    "kar",
    "diya",
    "gayi",
    "mushkil",
    "baaki",
    "abhi",
    "baad",
    "bata",
    "sakta",
    "hoon",
    "mein",
    "mujhe",
}


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[?.!,;:]+$", "", text).strip()
    return text


def _language(text: str) -> str:
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return (
        "hinglish"
        if len(tokens & HINGLISH_MARKERS) >= 2
        else "en"
    )


def _domain_scores(q: str) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)

    for domain, terms in DOMAIN_TERMS.items():
        for phrase, weight in terms.items():
            if phrase in q:
                scores[domain] += weight

    if any(
        x in q
        for x in (
            "partial payment",
            "half payment",
            "pay only part",
            "paid a little",
        )
    ):
        scores["repayment"] += 5.0

    payment_made = any(
        x in q
        for x in (
            "already paid",
            "payment kar diya",
            "payment cleared",
            "payment succeeded",
            "transfer is complete",
            "made the payment",
        )
    )

    posting_problem = any(
        x in q
        for x in (
            "not showing",
            "show nahi",
            "not updated",
            "update nahi hua",
            "cannot see",
            "can't see",
            "hasn't appeared",
            "not appeared",
            "not reflected",
            "system disagrees",
            "records disagree",
        )
    )

    if payment_made and posting_problem:
        scores["dispute"] += 6.0

    elif payment_made and "pay again" in q:
        scores["support"] += 6.0

    if (
        "wrong number" in q
        or "number belongs to someone else" in q
    ):
        scores["privacy"] += 5.0

    return dict(scores)


def _pick_domain(
    scores: dict[str, float],
) -> tuple[str | None, float]:

    if not scores:
        return None, 0.0

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    domain, best = ranked[0]

    second = (
        ranked[1][1]
        if len(ranked) > 1
        else 0.0
    )

    margin = best - second

    confidence = min(
        0.96,
        0.50
        + best * 0.055
        + max(0.0, margin) * 0.025,
    )

    return domain, confidence


def _is_ambiguous(
    q: str,
    domain_scores: dict[str, float],
) -> bool:

    if domain_scores:
        return False

    if any(
        re.fullmatch(pattern, q)
        for pattern in GENERIC_VAGUE_PATTERNS
    ):
        return True

    words = re.findall(
        r"[a-z0-9]+",
        q,
    )

    generic_words = {
        "help",
        "assist",
        "assistance",
        "issue",
        "problem",
        "situation",
        "this",
        "that",
        "it",
        "options",
        "next",
        "fix",
        "fixed",
        "account",
    }

    content = {
        word
        for word in words
        if len(word) > 2
    }

    if (
        len(words) <= 8
        and content
        and len(content - generic_words) <= 2
    ):
        return True

    return False


def _is_out_of_scope(
    q: str,
    domain_scores: dict[str, float],
) -> bool:

    if domain_scores:
        return False

    return any(
        term in q
        for term in OUT_OF_SCOPE_TERMS
    )


def _unsupported_account_action(q: str) -> bool:
    return any(
        re.search(pattern, q)
        for pattern in UNSUPPORTED_ACCOUNT_PATTERNS
    )


def _split_subqueries(
    question: str,
) -> list[str]:

    parts = re.split(
        r"\s*(?:;|\band also\b|\balso\b|\band\b|\bplus\b)\s*",
        question,
        flags=re.IGNORECASE,
    )

    cleaned = [
        part.strip(" ?.!,")
        for part in parts
        if len(part.strip(" ?.!,"))
        >= 4
    ]

    return (
        cleaned
        if len(cleaned) >= 2
        else []
    )


class QueryAnalyzer:

    def analyze(
        self,
        question: str,
        context=None,
    ) -> QueryAnalysis:

        q = _normalize(question)
        language = _language(q)

        if _unsupported_account_action(q):
            return QueryAnalysis(
                query_type=(
                    QueryType.PROCEDURAL_QUESTION
                ),
                domain="account_management",
                confidence=0.97,
                language=language,
            )

        scores = _domain_scores(q)

        if _is_out_of_scope(q, scores):
            return QueryAnalysis(
                query_type=QueryType.OUT_OF_SCOPE,
                domain=None,
                confidence=0.97,
                language=language,
            )

        if _is_ambiguous(q, scores):
            return QueryAnalysis(
                query_type=QueryType.AMBIGUOUS,
                domain=None,
                needs_clarification=True,
                clarification_reason=(
                    "The request does not identify "
                    "a specific supported policy "
                    "or account issue."
                ),
                confidence=0.95,
                language=language,
            )

        domain, domain_confidence = (
            _pick_domain(scores)
        )

        subqueries = (
            _split_subqueries(question)
        )

        meaningful_subqueries = []

        for subquery in subqueries:

            sub_scores = _domain_scores(
                _normalize(subquery)
            )

            if (
                sub_scores
                or len(subquery.split()) >= 4
            ):
                meaningful_subqueries.append(
                    subquery
                )

        is_multi = (
            len(meaningful_subqueries)
            >= 2
        )

        return QueryAnalysis(
            query_type=(
                QueryType.MULTI_PART
                if is_multi
                else QueryType.POLICY_QUESTION
            ),
            domain=domain,
            subqueries=(
                meaningful_subqueries
                if is_multi
                else []
            ),
            language=language,
            confidence=(
                domain_confidence
                if domain
                else 0.60
            ),
        )