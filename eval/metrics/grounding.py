def expected_fact_coverage(record, result):
    if not record.expected_facts:
        return 1.0
    text = (result.answer or "").lower()
    hits = sum(1 for fact in record.expected_facts if fact.lower() in text)
    return hits / len(record.expected_facts)
