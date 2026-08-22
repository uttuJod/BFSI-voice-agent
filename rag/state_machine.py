from .schemas import EvidenceVerdict, PipelineState

TRANSITIONS = {
    PipelineState.START: {PipelineState.ANALYZE_QUERY},
    PipelineState.ANALYZE_QUERY: {
        PipelineState.RETRIEVE,
        PipelineState.ASK_CLARIFICATION,
        PipelineState.ABSTAIN,
    },
    PipelineState.RETRIEVE: {PipelineState.EVALUATE_EVIDENCE},
    PipelineState.EVALUATE_EVIDENCE: {
        PipelineState.GENERATE_ANSWER,
        PipelineState.REWRITE_QUERY,
        PipelineState.DECOMPOSE_QUERY,
        PipelineState.RESOLVE_CONFLICT,
        PipelineState.ASK_CLARIFICATION,
        PipelineState.ABSTAIN,
    },
    PipelineState.REWRITE_QUERY: {PipelineState.RETRIEVE},
    PipelineState.DECOMPOSE_QUERY: {PipelineState.RETRIEVE},
    PipelineState.RESOLVE_CONFLICT: {
        PipelineState.GENERATE_ANSWER,
        PipelineState.ASK_CLARIFICATION,
        PipelineState.ABSTAIN,
    },
    PipelineState.GENERATE_ANSWER: {PipelineState.COMPLETE},
    PipelineState.ASK_CLARIFICATION: {PipelineState.COMPLETE},
    PipelineState.ABSTAIN: {PipelineState.COMPLETE},
}

def next_state_for_verdict(verdict: EvidenceVerdict) -> PipelineState:
    return {
        EvidenceVerdict.SUFFICIENT: PipelineState.GENERATE_ANSWER,
        EvidenceVerdict.LOW_RELEVANCE: PipelineState.REWRITE_QUERY,
        EvidenceVerdict.INSUFFICIENT: PipelineState.DECOMPOSE_QUERY,
        EvidenceVerdict.CONFLICT: PipelineState.RESOLVE_CONFLICT,
        EvidenceVerdict.AMBIGUOUS: PipelineState.ASK_CLARIFICATION,
        EvidenceVerdict.OUT_OF_SCOPE: PipelineState.ABSTAIN,
    }[verdict]
