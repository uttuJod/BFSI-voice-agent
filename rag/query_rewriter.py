from __future__ import annotations


class QueryRewriter:

    DOMAIN_EXPANSIONS = {
        "hardship": (
            "financial hardship reduced income "
            "unemployment temporary difficulty assistance"
        ),

        "collections": (
            "collections overdue grace period due date "
            "promise-to-pay current policy"
        ),

        "repayment": (
            "repayment partial payment instalment "
            "structured repayment arrangement"
        ),

        "privacy": (
            "privacy third party disclosure "
            "account information consent"
        ),

        "dispute": (
            "payment dispute transaction posted "
            "missing payment records mismatch investigation"
        ),

        "identity": (
            "identity verification authentication "
            "account information security"
        ),

        "escalation": (
            "human escalation specialist "
            "callback support handoff"
        ),

        "support": (
            "customer support FAQ support hours "
            "payment already made posting status"
        ),
    }

    GENERIC_EXPANSION = (
        "BFSI customer support policy procedure "
        "eligibility current active policy"
    )

    def rewrite(
        self,
        question: str,
        analysis,
        iteration: int,
    ) -> str:

        if (
            analysis.domain
            and analysis.confidence
            >= 0.78
        ):

            suffix = (
                self.DOMAIN_EXPANSIONS.get(
                    analysis.domain,
                    self.GENERIC_EXPANSION,
                )
            )

        else:

            suffix = (
                self.GENERIC_EXPANSION
            )

        if iteration >= 2:

            suffix += (
                " authoritative applicable rule "
                "customer request"
            )

        return (
            f"{question.strip()} {suffix}"
        ).strip()

    def decompose(
        self,
        question: str,
        analysis,
    ) -> list[str]:

        if analysis.subqueries:
            return analysis.subqueries

        return [
            self.rewrite(
                question,
                analysis,
                1,
            )
        ]