def retrieval_recall_at_k(record, result):
    if not record.expected_sources:
        return 1.0
    retrieved = {c.document_id for c in result.retrieved_chunks}
    expected = set(record.expected_sources)
    return len(retrieved & expected) / len(expected)
