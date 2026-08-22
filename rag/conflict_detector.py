import re
from collections import defaultdict
from .schemas import (
    ConflictAnalysis, ConflictItem, ConflictResolutionStatus,
    CorrectionAction, DocumentStatus,
)

PATTERNS = {
    "grace_period_days": re.compile(r"grace period(?: is| of)?\s+(\d+)\s+days?", re.I),
    "callback_hours": re.compile(r"callback.*?(\d+)\s+hours?", re.I),
}

class ConflictDetector:
    def detect(self, chunks):
        claims = defaultdict(list)
        for c in chunks:
            for key, pat in PATTERNS.items():
                m = pat.search(c.text)
                if m:
                    claims[key].append(ConflictItem(
                        claim_key=key,
                        document_id=c.document_id,
                        chunk_id=c.chunk_id,
                        value=m.group(1),
                        version=c.metadata.version,
                        effective_date=c.metadata.effective_date,
                        status=c.metadata.status,
                    ))
        conflicting = []
        for items in claims.values():
            if len({x.value for x in items}) > 1:
                conflicting.extend(items)

        if not conflicting:
            return ConflictAnalysis()

        active = [x for x in conflicting if x.status == DocumentStatus.ACTIVE]
        superseded = [x for x in conflicting if x.status == DocumentStatus.SUPERSEDED]

        if len(active) == 1 and superseded:
            return ConflictAnalysis(
                conflict_detected=True,
                conflicting_claims=conflicting,
                resolution_status=ConflictResolutionStatus.RESOLVED,
                preferred_document_id=active[0].document_id,
                resolution_reason="One conflicting source is active while the other is superseded.",
                recommended_action=CorrectionAction.CONFLICT_RESOLUTION,
            )

        if active:
            latest = sorted(
                active,
                key=lambda x: (x.effective_date or __import__("datetime").date.min, x.version or 0),
                reverse=True,
            )
            if len(latest) >= 2 and latest[0].effective_date != latest[1].effective_date:
                return ConflictAnalysis(
                    conflict_detected=True,
                    conflicting_claims=conflicting,
                    resolution_status=ConflictResolutionStatus.RESOLVED,
                    preferred_document_id=latest[0].document_id,
                    resolution_reason="Preferred the active source with the latest effective date.",
                    recommended_action=CorrectionAction.CONFLICT_RESOLUTION,
                )

        return ConflictAnalysis(
            conflict_detected=True,
            conflicting_claims=conflicting,
            resolution_status=ConflictResolutionStatus.UNRESOLVED,
            resolution_reason="Conflicting evidence could not be resolved safely from metadata.",
            recommended_action=CorrectionAction.ABSTAIN,
        )
