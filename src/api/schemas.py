"""
Pydantic v2 schemas for MemCore:
- Working Memory turns
- Bi-temporal Knowledge Graph Triplets & Edges
- Episodic Events
- Procedural Playbooks
- Fast-Path Chat Requests / Responses & Context Telemetry
- Time-Travel queries
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class TurnRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScratchpadState(BaseModel):
    user_id: str
    session_id: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TripletFact(BaseModel):
    subject: str = Field(..., description="Subject entity name, e.g. 'backend', 'user'")
    predicate: str = Field(..., description="Relation predicate, e.g. 'USES_DB', 'DEPLOYED_ON'")
    object: str = Field(..., description="Object entity or value, e.g. 'PostgreSQL 16', 'Kubernetes'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.subject.strip().lower()}:{self.predicate.strip().upper()}"

    def to_readable_sentence(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


class GraphEdgeRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: Optional[datetime] = None
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    superseded_by: Optional[str] = None
    source_session_id: Optional[str] = None
    source_turn_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @property
    def is_active(self) -> bool:
        return self.valid_until is None

    def as_dict_repr(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "is_active": self.is_active,
            "superseded_by": self.superseded_by,
        }


class EpisodicEventRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str
    raw_context: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    turn_range: Optional[List[int]] = None
    embedding: Optional[List[float]] = None


class ProceduralRule(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    rule: str
    category: Literal["coding_style", "architecture", "formatting", "workflow", "negative_constraint", "general"] = "general"
    trigger_pattern: str = ""
    confidence: float = 0.95
    occurrences: int = 1
    learned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_applied_at: Optional[datetime] = None
    source_correction: Optional[str] = None


class ProceduralPlaybookRecord(BaseModel):
    user_id: str
    rules: List[ProceduralRule] = Field(default_factory=list)
    system_patch: str = ""
    version: int = 1
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExtractionResult(BaseModel):
    session_summary: str
    facts: List[TripletFact] = Field(default_factory=list)
    user_corrections: List[Dict[str, Any]] = Field(default_factory=list)
    rule_candidates: List[ProceduralRule] = Field(default_factory=list)


class ConflictResolutionDiff(BaseModel):
    action: Literal["NOOP", "CONTRADICTION", "NEW_FACT"]
    triplet: TripletFact
    previous_edge: Optional[GraphEdgeRecord] = None
    resulting_edge: Optional[GraphEdgeRecord] = None
    description: str


class ConsolidationSummary(BaseModel):
    user_id: str
    session_id: str
    turns_processed: int
    facts_extracted: int
    contradictions_resolved: int
    noops: int
    new_facts_added: int
    rules_distilled: int
    duration_ms: float
    diffs: List[ConflictResolutionDiff] = Field(default_factory=list)
    rules: List[ProceduralRule] = Field(default_factory=list)


class ContextTelemetry(BaseModel):
    tokens_used: int
    token_budget: int
    latency_ms: float
    breakdown: Dict[str, int] = Field(
        default_factory=lambda: {
            "playbook_tokens": 0,
            "active_facts_tokens": 0,
            "episodic_tokens": 0,
            "working_memory_tokens": 0,
        }
    )
    pruned_items_count: int = 0
    active_facts_count: int = 0
    active_facts_summary: List[str] = Field(default_factory=list)
    playbook_rules_applied: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"
    user_id: str = "default_user"
    stream: bool = False
    as_of_time: Optional[datetime] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    user_id: str
    context_telemetry: ContextTelemetry
    consolidated_sleep_triggered: bool = False


class SleepTriggerRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = "default_session"
    force: bool = False


class SleepTriggerResponse(BaseModel):
    status: str
    message: str
    task_id: Optional[str] = None
    summary: Optional[ConsolidationSummary] = None


class TimeTravelQuery(BaseModel):
    user_id: str = "default_user"
    as_of_time: datetime
    subject: Optional[str] = None
    predicate: Optional[str] = None


class TimeTravelResult(BaseModel):
    user_id: str
    as_of_time: datetime
    active_facts: List[GraphEdgeRecord]
    expired_facts: List[GraphEdgeRecord]
    explanation: str


class SeedScenarioRequest(BaseModel):
    scenario: Literal["temporal_contradiction", "procedural_evolution", "combined"] = "temporal_contradiction"
    user_id: str = "demo_user"
    session_id: str = "demo_session"
