from pydantic import BaseModel, Field

class EvalRecord(BaseModel):
    id: str
    question: str
    category: str
    expected_behavior: str
    answerable: bool
    expected_sources: list[str] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)

class EvalOutcome(BaseModel):
    id: str
    system: str
    passed: bool
    failure_categories: list[str] = Field(default_factory=list)
    latency_ms: float
