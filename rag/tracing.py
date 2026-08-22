from .schemas import StateTransition, PipelineState

class TraceRecorder:
    def __init__(self):
        self.transitions = []

    def add(self, from_state: PipelineState, to_state: PipelineState, reason: str):
        self.transitions.append(StateTransition(
            from_state=from_state,
            to_state=to_state,
            reason=reason,
        ))
