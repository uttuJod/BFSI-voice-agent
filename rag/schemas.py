from __future__ import annotations
from datetime import date
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

class QueryType(str, Enum):
    POLICY_QUESTION = "policy_question"
    COMPARISON = "comparison"
    PROCEDURAL_QUESTION = "procedural_question"
    ELIGIBILITY_QUESTION = "eligibility_question"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS = "ambiguous"
    MULTI_PART = "multi_part"

class EvidenceVerdict(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    LOW_RELEVANCE = "LOW_RELEVANCE"
    INSUFFICIENT = "INSUFFICIENT"
    CONFLICT = "CONFLICT"
    AMBIGUOUS = "AMBIGUOUS"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"

class CorrectionAction(str, Enum):
    NONE = "NONE"
    QUERY_REWRITE = "QUERY_REWRITE"
    QUERY_DECOMPOSITION = "QUERY_DECOMPOSITION"
    BROADEN_RETRIEVAL = "BROADEN_RETRIEVAL"
    METADATA_FILTER = "METADATA_FILTER"
    CONFLICT_RESOLUTION = "CONFLICT_RESOLUTION"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    ABSTAIN = "ABSTAIN"

class DocumentStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DRAFT = "draft"
    ARCHIVED = "archived"
    UNKNOWN = "unknown"

class ConflictResolutionStatus(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"

class PipelineState(str, Enum):
    START = "START"
    ANALYZE_QUERY = "ANALYZE_QUERY"
    RETRIEVE = "RETRIEVE"
    EVALUATE_EVIDENCE = "EVALUATE_EVIDENCE"
    REWRITE_QUERY = "REWRITE_QUERY"
    DECOMPOSE_QUERY = "DECOMPOSE_QUERY"
    RESOLVE_CONFLICT = "RESOLVE_CONFLICT"
    GENERATE_ANSWER = "GENERATE_ANSWER"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    ABSTAIN = "ABSTAIN"
    COMPLETE = "COMPLETE"

class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    document_id: str
    title: str
    version: int | None = None
    effective_date: date | None = None
    status: DocumentStatus = DocumentStatus.UNKNOWN
    domain: str | None = None
    source_path: str | None = None
    language: str = "en"
    tags: list[str] = Field(default_factory=list)

class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: DocumentMetadata
    chunk_index: int = Field(ge=0)

class QueryAnalysis(BaseModel):
    query_type: QueryType
    domain: str | None = None
    entities: dict[str, Any] = Field(default_factory=dict)
    needs_clarification: bool = False
    clarification_reason: str | None = None
    subqueries: list[str] = Field(default_factory=list)
    language: str = "en"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class SearchFilters(BaseModel):
    domain: str | None = None
    status: DocumentStatus | None = None
    document_ids: list[str] | None = None
    effective_on_or_before: date | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    filters: SearchFilters | None = None

class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: DocumentMetadata
    rank: int = Field(ge=1)

class EvidenceEvaluation(BaseModel):
    verdict: EvidenceVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    conflict_detected: bool = False
    missing_information: list[str] = Field(default_factory=list)
    reason: str
    supported_document_ids: list[str] = Field(default_factory=list)

class ConflictItem(BaseModel):
    claim_key: str
    document_id: str
    chunk_id: str
    value: str
    version: int | None = None
    effective_date: date | None = None
    status: DocumentStatus = DocumentStatus.UNKNOWN

class ConflictAnalysis(BaseModel):
    conflict_detected: bool = False
    conflicting_claims: list[ConflictItem] = Field(default_factory=list)
    resolution_status: ConflictResolutionStatus = ConflictResolutionStatus.NOT_APPLICABLE
    preferred_document_id: str | None = None
    resolution_reason: str | None = None
    recommended_action: CorrectionAction = CorrectionAction.NONE

class RetrievalTrace(BaseModel):
    iteration: int = Field(ge=1)
    query: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    verdict: EvidenceVerdict | None = None
    action: CorrectionAction = CorrectionAction.NONE
    reason: str | None = None

class StateTransition(BaseModel):
    from_state: PipelineState
    to_state: PipelineState
    reason: str

class Citation(BaseModel):
    document_id: str
    chunk_id: str
    title: str | None = None
    version: int | None = None
    effective_date: date | None = None

class LatencyBreakdown(BaseModel):
    analysis_ms: float = Field(default=0.0, ge=0.0)
    retrieval_ms: float = Field(default=0.0, ge=0.0)
    evaluation_ms: float = Field(default=0.0, ge=0.0)
    generation_ms: float = Field(default=0.0, ge=0.0)
    total_ms: float = Field(default=0.0, ge=0.0)

class RAGResult(BaseModel):
    answer: str | None = None
    answerable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    verdict: EvidenceVerdict
    retrieval_iterations: int = Field(default=0, ge=0)
    query_history: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    conflict_detected: bool = False
    conflict_resolution: ConflictAnalysis | None = None
    correction_actions: list[CorrectionAction] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    clarification_question: str | None = None
    abstention_reason: str | None = None
    trace: list[RetrievalTrace] = Field(default_factory=list)
    state_transitions: list[StateTransition] = Field(default_factory=list)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)

class RAGContext(BaseModel):
    intent: str | None = None
    language: str = "en"
    customer_context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
