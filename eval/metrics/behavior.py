def behavior_correct(record, result):
    if record.expected_behavior == "answer":
        return bool(result.answerable and result.answer)
    if record.expected_behavior == "abstain":
        return not result.answerable and result.verdict.value in {"INSUFFICIENT", "OUT_OF_SCOPE", "CONFLICT"}
    if record.expected_behavior == "clarify":
        return result.verdict.value == "AMBIGUOUS" and bool(result.clarification_question)
    if record.expected_behavior == "conflict":
        return bool(result.conflict_detected)
    return False

def source_attribution_correct(record, result):
    if not record.expected_sources:
        return True
    actual = {c.document_id for c in result.citations}
    return set(record.expected_sources).issubset(actual)

def forbidden_claim_violation(record, result):
    text = (result.answer or "").lower()
    return any(x.lower() in text for x in record.forbidden_claims)
