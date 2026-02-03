"""
pydantic_models.py

All Pydantic models, enums, and type definitions for the LLM confidence chat loop.
This file contains the data structures used throughout the controller system.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# =============================================================================
# Enums / Literals
# =============================================================================

ArtifactType = Literal["text", "screenshot", "logs", "file", "recording", "audio"]
ActionType = Literal["ASK_EVIDENCE", "EXEC_STEP", "SWITCH_LANE", "ESCALATE", "RESOLVE"]
RiskLevel = Literal["low", "medium", "high"]
EvidenceActionKind = Literal["question", "artifact_request", "confirmation"]
StepKind = Literal["instruction", "check", "decision", "artifact", "escalation", "rollback", "note"]
FactType = Literal["string", "boolean", "enum", "number", "list_string", "json"]
PolicyReasonCode = Literal[
    "THRESHOLD_NOT_MET",
    "CROSS_THRESHOLD_COMMIT",
    "HIGH_INFO_GAIN_EVIDENCE",
    "EXEC_NEXT_STEP",
    "LANE_DOMINANT_MARGIN_SWITCH",
    "LANE_SWITCH_BLOCKED_SMALL_MARGIN",
    "LANE_SWITCH_BLOCKED_LOW_CONF",
    "BUDGET_REACHED_SAFE_COMMIT",
    "BUDGET_REACHED_ESCALATE",
    "FAILED_STEP_REEVALUATE",
    "RESOLVED_CONFIRMED",
    "TERMINAL_VALIDATION_RESOLVED",
    "TERMINAL_VALIDATION_ESCALATE",
    "DETERMINISTIC_STEP_TRANSITION",
]


class Comparator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    IN = "in"
    NIN = "nin"
    CONTAINS = "contains"
    REGEX = "regex"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    GTE = "gte"
    LTE = "lte"


class Phase(str, Enum):
    TRIAGE = "triage"
    EXECUTE = "execute"


SlotSafety = Literal["safe", "sensitive", "blocked"]


# =============================================================================
# Helper functions for validation
# =============================================================================

def _canon_key(v: str) -> str:
    v = (v or "").strip()
    v = v.replace(" ", "_")
    v = re.sub(r"[^a-zA-Z0-9_\-:.]", "", v)
    return v.lower()


_ALLOWED_LHS_PREFIX = ("fact.", "slot.", "artifact.", "ctx.", "belief.")


# =============================================================================
# Conditions / Constraint sets
# =============================================================================

class Condition(BaseModel):
    lhs: str = Field(..., description="fact.* | slot.* | artifact.* | ctx.* | belief.*")
    op: Comparator
    rhs: Optional[Any] = None

    @field_validator("lhs")
    @classmethod
    def _lhs_prefix(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(_ALLOWED_LHS_PREFIX):
            raise ValueError(f"lhs must start with one of: {_ALLOWED_LHS_PREFIX}")
        if not re.match(r"^[a-zA-Z0-9_.:\-]+$", v):
            raise ValueError("lhs must be path-safe (letters/digits/._:-)")
        return v


class ConstraintSet(BaseModel):
    all_of: List[Condition] = Field(default_factory=list)
    any_of: List[Condition] = Field(default_factory=list)
    none_of: List[Condition] = Field(default_factory=list)


# =============================================================================
# Facts and extraction models
# =============================================================================

class FactSpec(BaseModel):
    fact: str = Field(..., description="Stable key, e.g. 'os', 'app', 'error_code', 'vpn_on'")
    description: str = ""
    type: FactType = "string"
    allowed_values: List[str] = Field(default_factory=list)
    extraction_hint: Optional[str] = None
    required_for_certainty: bool = False

    @field_validator("fact")
    @classmethod
    def _fact_key(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[a-zA-Z0-9_\-:.]+$", v):
            raise ValueError("fact must be url-safe-ish (letters/digits/_-:.)")
        return _canon_key(v)


class EvidenceItem(BaseModel):
    evidence_id: str
    type: ArtifactType = "text"
    summary: str = Field(..., description="Short summary; avoid raw PII here")
    sha256: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_id")
    @classmethod
    def _eid_safe(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[a-zA-Z0-9_\-:.]+$", v):
            raise ValueError("evidence_id must be url-safe-ish")
        return v


class ExtractedFact(BaseModel):
    fact: str
    value: Any = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    source: Literal["chat", "voice", "artifact", "system"] = "chat"
    evidence_id: Optional[str] = None

    @field_validator("fact")
    @classmethod
    def _fact_key(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[a-zA-Z0-9_\-:.]+$", v):
            raise ValueError("fact must be url-safe-ish")
        return _canon_key(v)


class ExtractionTrace(BaseModel):
    received: List[EvidenceItem] = Field(default_factory=list)
    extracted_facts: List[ExtractedFact] = Field(default_factory=list)
    missing_fact_keys: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ExtractorOutput(BaseModel):
    extraction: ExtractionTrace


# =============================================================================
# Evidence actions
# =============================================================================

def _opt_str(v: Any) -> Optional[str]:
    """Coerce null/empty/"None"/"null" into None; otherwise return a clean string."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"none", "null"}:
            return None
        return s
    return str(v)


class EvidenceRequest(BaseModel):
    type: ArtifactType = "text"
    prompt: str = Field(..., min_length=1)
    targets: List[str] = Field(default_factory=list, description="Fact keys expected")
    examples: List[str] = Field(default_factory=list)
    redact_hints: List[str] = Field(default_factory=list)
    parse_hint: Optional[str] = None

    @field_validator("targets")
    @classmethod
    def _targets_keys(cls, v: List[str]) -> List[str]:
        out: List[str] = []
        for k in v or []:
            kk = _canon_key(k)
            if not kk or not re.match(r"^[a-zA-Z0-9_\-:.]+$", kk):
                raise ValueError(f"Invalid target fact key: {k}")
            out.append(kk)
        return out

    @field_validator("parse_hint", mode="before")
    @classmethod
    def _coerce_parse_hint(cls, v: Any) -> Optional[str]:
        return _opt_str(v)


class EvidenceAction(BaseModel):
    action_id: str
    kind: EvidenceActionKind = "question"
    intent: str = Field(..., description="Why we ask; what it disambiguates")
    request: EvidenceRequest
    cost: Literal["low", "med", "high"] = "low"
    discriminates_lanes: List[str] = Field(default_factory=list)
    expected_info_gain: float = Field(0.0, ge=0.0, le=1.0)

    @field_validator("action_id")
    @classmethod
    def _id_safe(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[a-zA-Z0-9_\-:.]+$", v):
            raise ValueError("action_id must be url-safe-ish")
        return v


# =============================================================================
# Procedure lane & steps
# =============================================================================

class RetryPolicy(BaseModel):
    max_attempts: int = Field(1, ge=1, le=10)
    backoff_ms: int = Field(0, ge=0, le=120_000)
    retry_on: List[str] = Field(default_factory=list)


class ValidationRule(BaseModel):
    rule_id: str
    description: str = ""
    when: ConstraintSet = Field(default_factory=ConstraintSet)
    effect: Literal["increase_belief", "decrease_belief", "terminal_resolved", "terminal_escalate"] = "increase_belief"
    weight: float = Field(1.0, ge=0.0, le=10.0)

    @field_validator("rule_id")
    @classmethod
    def _rid_safe(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[a-zA-Z0-9_\-:.]+$", v):
            raise ValueError("rule_id must be url-safe-ish")
        return v


class TransitionEdge(BaseModel):
    to_step_id: str
    when: ConstraintSet = Field(default_factory=ConstraintSet)
    belief_delta: float = Field(0.0, ge=-1.0, le=1.0)
    rationale: Optional[str] = None


class LookaheadHint(BaseModel):
    unlocks_facts: List[str] = Field(default_factory=list)
    info_gain_hint: Optional[str] = None
    cost: Literal["low", "med", "high"] = "low"


class ProcedureStep(BaseModel):
    step_id: str
    kind: StepKind = "instruction"
    title: Optional[str] = None
    instruction: str = Field(..., min_length=1)
    how_to_reach: List[str] = Field(default_factory=list)
    evidence_actions: List[EvidenceAction] = Field(default_factory=list)
    validations: List[ValidationRule] = Field(default_factory=list)
    pass_when: ConstraintSet = Field(default_factory=ConstraintSet)
    fail_when: ConstraintSet = Field(default_factory=ConstraintSet)
    on_pass: List[TransitionEdge] = Field(default_factory=list)
    on_fail: List[TransitionEdge] = Field(default_factory=list)
    risk: RiskLevel = "low"
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    idempotency_key: Optional[str] = None
    lookahead: LookaheadHint = Field(default_factory=LookaheadHint)
    requires_admin: bool = False
    optional: bool = False
    safety_notes: Optional[str] = None
    expected_observation: Optional[str] = None

    @field_validator("step_id")
    @classmethod
    def _sid_safe(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[a-zA-Z0-9_\-:.]+$", v):
            raise ValueError("step_id must be url-safe-ish")
        return v

    @field_validator("safety_notes", mode="before")
    @classmethod
    def _coerce_safety_notes(cls, v: Any) -> Optional[str]:
        return _opt_str(v)

    @field_validator("expected_observation", mode="before")
    @classmethod
    def _coerce_expected_observation(cls, v: Any) -> Optional[str]:
        return _opt_str(v)


class SlotSafetyEntry(BaseModel):
    slot: str
    safety: SlotSafety


class ProcedureLanePrecondition(BaseModel):
    slot: str
    values: List[str] = Field(default_factory=list)


class SlotQuestion(BaseModel):
    slot: str
    question: str


class ArtifactRequest(BaseModel):
    type: Literal["screenshot", "text", "logs", "file", "audio", "recording"] = "text"
    prompt: str
    when: Optional[str] = None
    redact_hints: List[str] = Field(default_factory=list)
    parse_hint: Optional[str] = None

    @field_validator("when", mode="before")
    @classmethod
    def _coerce_when(cls, v: Any) -> Optional[str]:
        return _opt_str(v)

    @field_validator("parse_hint", mode="before")
    @classmethod
    def _coerce_parse_hint(cls, v: Any) -> Optional[str]:
        return _opt_str(v)


class ProcedureLane(BaseModel):
    """
    Matches generated JSONL.
    Compatibility:
    - `procedure_id` in JSONL is aliased into `lane_id` here
    - `slot_safety` can be dict or list[{slot,safety}] and is normalized to dict
    
    HYBRID RETRIEVAL (2024):
    - fact_keywords: BM25 lexical matching (exact terms, error codes)
    - fact_embeddings: Semantic similarity (concepts, paraphrasing)
    - fact_descriptions: Cross-encoder reranking (high-accuracy disambiguation)
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    lane_id: str = Field(alias="procedure_id")
    doc_id: str = ""
    doc_title: str = ""
    title: str = ""
    summary: str = ""
    preconditions: List[ProcedureLanePrecondition] = Field(default_factory=list)
    fact_specs: List[FactSpec] = Field(default_factory=list)
    required_slots: List[str] = Field(default_factory=list)
    signatures: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    entry_step_id: str = ""
    steps: List[ProcedureStep] = Field(default_factory=list)
    verify_prompts: List[str] = Field(default_factory=list)
    escalate_requirements: List[str] = Field(default_factory=list)
    risk: RiskLevel = "medium"
    ask_slots: List[str] = Field(default_factory=list)
    slot_priority: List[str] = Field(default_factory=list)
    slot_questions: List[SlotQuestion] = Field(default_factory=list)
    slot_safety: Union[Dict[str, SlotSafety], List[SlotSafetyEntry]] = Field(default_factory=dict)
    artifact_requests: List[ArtifactRequest] = Field(default_factory=list)
    ask_if_uncertain: List[str] = Field(default_factory=list)
    source: Dict[str, Any] = Field(default_factory=dict, alias="_source")
    
    # Hybrid retrieval indices (for BM25 + semantic + cross-encoder matching)
    fact_keywords: List[str] = Field(
        default_factory=list, 
        description="BM25 keywords for lexical retrieval (exact terms, error codes, product names)"
    )
    fact_embeddings: List[List[float]] = Field(
        default_factory=list, 
        description="Dense embeddings for semantic search (handles synonyms, paraphrasing, concepts)"
    )
    fact_descriptions: List[str] = Field(
        default_factory=list, 
        description="Text descriptions for cross-encoder reranking (high-accuracy disambiguation)"
    )

    @model_validator(mode="after")
    def _normalize_lane(self) -> "ProcedureLane":
        # normalize slot_safety to dict
        if isinstance(self.slot_safety, list):
            self.slot_safety = {e.slot: e.safety for e in self.slot_safety}

        # normalize fact key names
        self.fact_specs = [
            FactSpec(**{**fs.model_dump(), "fact": _canon_key(fs.fact)}) 
            for fs in (self.fact_specs or [])
        ]

        # if entry_step_id missing but we have steps, default to first
        if not self.entry_step_id and self.steps:
            self.entry_step_id = self.steps[0].step_id
        return self

    def step_map(self) -> Dict[str, ProcedureStep]:
        return {s.step_id: s for s in (self.steps or [])}


# =============================================================================
# Belief / budgets / switching
# =============================================================================

class BeliefItem(BaseModel):
    key: str
    belief: float = Field(..., ge=0.0, le=1.0)


class BeliefState(BaseModel):
    items: List[BeliefItem] = Field(default_factory=list)
    temperature: float = Field(0.7, ge=0.01, le=5.0)

    def as_dict(self) -> Dict[str, float]:
        return {it.key: it.belief for it in self.items}

    def topk(self, k: int = 5) -> List[BeliefItem]:
        return sorted(self.items, key=lambda x: x.belief, reverse=True)[:k]


class BudgetPolicy(BaseModel):
    max_precommit_questions: int = Field(3, ge=0, le=20)
    max_precommit_artifacts: int = Field(1, ge=0, le=10)
    max_steps_before_reconsider: int = Field(3, ge=1, le=50)
    max_failed_steps_before_escalate: int = Field(2, ge=0, le=20)
    max_total_questions: int = Field(8, ge=0, le=100)
    max_total_turns: int = Field(20, ge=1, le=500)
    max_user_effort_score: float = Field(10.0, ge=0.0, le=100.0)


class SwitchingPolicy(BaseModel):
    enter_lane_threshold: float = Field(0.60, ge=0.0, le=1.0)
    strong_lane_threshold: float = Field(0.75, ge=0.0, le=1.0)
    switch_margin: float = Field(0.20, ge=0.0, le=1.0)
    require_consecutive_turns: int = Field(2, ge=1, le=10)
    max_progress_switch_cost: float = Field(0.20, ge=0.0, le=1.0)


class RuntimeCounters(BaseModel):
    precommit_questions: int = 0
    precommit_artifacts: int = 0
    total_questions: int = 0
    total_turns: int = 0
    failed_steps: int = 0
    steps_executed_in_lane: int = 0
    user_effort_score: float = 0.0


class ControllerState(BaseModel):
    current_lane_id: Optional[str] = None
    current_step_id: Optional[str] = None
    phase: Phase = Phase.TRIAGE
    belief: BeliefState = Field(default_factory=BeliefState)
    budgets: BudgetPolicy = Field(default_factory=BudgetPolicy)
    switching: SwitchingPolicy = Field(default_factory=SwitchingPolicy)
    counters: RuntimeCounters = Field(default_factory=RuntimeCounters)
    switch_vote_lane_id: Optional[str] = None
    switch_vote_count: int = 0
    asked_evidence_action_ids: List[str] = Field(default_factory=list)
    known_facts: Dict[str, ExtractedFact] = Field(default_factory=dict)


# =============================================================================
# Debug / traces
# =============================================================================

class BeliefDelta(BaseModel):
    key: str
    before: float = Field(..., ge=0.0, le=1.0)
    after: float = Field(..., ge=0.0, le=1.0)
    reason: str


class BeliefTrace(BaseModel):
    topk_before: List[BeliefItem] = Field(default_factory=list)
    topk_after: List[BeliefItem] = Field(default_factory=list)
    deltas: List[BeliefDelta] = Field(default_factory=list)
    threshold_crossed: bool = False
    threshold_value: float = Field(0.0, ge=0.0, le=1.0)
    current_lane_id: Optional[str] = None
    candidate_lane_id: Optional[str] = None
    belief_gap: Optional[float] = Field(None, ge=-1.0, le=1.0)
    consecutive_switch_votes: int = Field(0, ge=0, le=100)
    phase: str = "triage"


class StepTrace(BaseModel):
    lane_id: str
    step_id: str
    action_taken: ActionType
    asked_evidence_action_ids: List[str] = Field(default_factory=list)
    validation_results: List[Tuple[str, bool]] = Field(default_factory=list)
    transition_taken: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class PolicyTrace(BaseModel):
    selected_action: ActionType
    reason_code: PolicyReasonCode
    rationale: str
    selected_lane_id: Optional[str] = None
    selected_step_id: Optional[str] = None
    selected_evidence_action_id: Optional[str] = None


class TurnTrace(BaseModel):
    turn_id: str
    extraction: ExtractionTrace = Field(default_factory=ExtractionTrace)
    belief: BeliefTrace = Field(default_factory=BeliefTrace)
    policy: PolicyTrace
    step: Optional[StepTrace] = None
    warnings: List[str] = Field(default_factory=list)

    @field_validator("turn_id")
    @classmethod
    def _turn_id_safe(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[a-zA-Z0-9_\-:.]+$", v):
            raise ValueError("turn_id must be url-safe-ish")
        return v


# =============================================================================
# LLM proposal models
# =============================================================================

class BeliefDeltaProposal(BaseModel):
    key: str = Field(..., description="lane_id")
    delta: float = Field(..., ge=-1.0, le=1.0)
    reason: str


class NextActionProposal(BaseModel):
    type: ActionType
    reason_code: PolicyReasonCode
    rationale: str
    lane_id: Optional[str] = None
    step_id: Optional[str] = None
    evidence_action_id: Optional[str] = None


class ControllerProposal(BaseModel):
    belief_deltas: List[BeliefDeltaProposal] = Field(default_factory=list)
    next_action: NextActionProposal


# =============================================================================
# Step evaluation result
# =============================================================================

class StepEvalResult(BaseModel):
    terminal: Optional[Literal["resolved", "escalate"]] = None
    validation_hits: List[Tuple[str, bool]] = Field(default_factory=list)
    belief_delta: float = 0.0
    belief_reason: str = ""
    passed: Optional[bool] = None
    failed: Optional[bool] = None
    transition_to: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


# =============================================================================
# Agent turn result
# =============================================================================

class AgentTurnResult(BaseModel):
    response_text: str
    state: ControllerState
    trace: TurnTrace