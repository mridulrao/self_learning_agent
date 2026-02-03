#!/usr/bin/env python3
"""
state_schema.py

State schema for an IT-support uncertainty-to-certainty agent.

Coherent with:
- procedure_compiler.py outputs (ProcedureNode JSONL)
- belief.py deterministic scoring outputs (belief + uncertainty)

Includes:
1) Evidence
   - user signal (chat/voice)
   - extractions from signal (slots + confidence, signatures, entities)
   - retrieval candidates (lightweight references + scores)
2) Belief
   - procedure hypotheses with probability + explainable score breakdown
3) Uncertainty
   - entropy, top1/top2/margin
   - coverage for top hypothesis, conflicts, signature strength

Dependencies:
  pip install pydantic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# -----------------------------
# 1) Evidence
# -----------------------------

SignalType = Literal["chat", "voice"]
SlotConf = Literal["unknown", "low", "medium", "high"]


class UserSignal(BaseModel):
    """
    Raw user input for the current turn.
    """
    type: SignalType
    text: str = Field(..., description="Chat text or voice transcript for this turn")
    language: Optional[str] = Field(None, description="BCP-47 tag if available, e.g., en-US")
    timestamp_ms: Optional[int] = Field(None, description="Client timestamp in ms")
    audio_uri: Optional[str] = Field(None, description="Pointer/URI to audio blob (if any)")


class ExtractedField(BaseModel):
    """
    Generic extracted field with confidence and provenance.
    Use this for slots, entities, signatures, etc.
    """
    name: str = Field(..., description="Field name, e.g., 'os', 'product', 'error_signature'")
    value: Optional[str] = Field(None, description="Extracted value (stringified)")
    conf: SlotConf = Field("unknown", description="Confidence bucket")
    source: Literal["user", "tool", "llm", "inferred"] = Field("llm", description="Where this came from")
    evidence_snippet: Optional[str] = Field(None, description="Short snippet supporting extraction")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("ExtractedField.name must be non-empty")
        return v


class ExtractionBundle(BaseModel):
    """
    Structured output of your 'frame extractor' LLM (or rules).
    """
    slots: List[ExtractedField] = Field(default_factory=list, description="Slot-like extracted fields")
    signatures: List[ExtractedField] = Field(default_factory=list, description="Error signatures/codes/messages")
    entities: List[ExtractedField] = Field(default_factory=list, description="Named entities relevant to IT support")
    notes: List[str] = Field(default_factory=list, description="Optional extraction notes or ambiguities")


class RetrievalCandidate(BaseModel):
    """
    Output from retriever BEFORE belief scoring.

    NOTE:
    - This is intentionally lightweight (does not embed the full ProcedureNode).
    - The controller typically uses procedure_id to fetch the full node object/dict,
      then calls belief.build_belief(frame, [(node, retrieval_score), ...]).
    """
    procedure_id: str
    retrieval_score: float
    doc_id: Optional[str] = None
    doc_title: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    risk: Optional[Literal["low", "medium", "high"]] = None


class Evidence(BaseModel):
    user_signal: UserSignal
    extractions: ExtractionBundle = Field(default_factory=ExtractionBundle)
    artifacts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Screenshots/logs/tool outputs; keep opaque for V1"
    )
    retrieval_candidates: List[RetrievalCandidate] = Field(
        default_factory=list,
        description="Top-K retrieved candidates BEFORE belief scoring"
    )


# -----------------------------
# 2) Belief object (matches belief.py)
# -----------------------------

RiskLevel = Literal["low", "medium", "high"]
SigStrength = Literal["none", "weak", "regex", "exact"]


class ScoreBreakdown(BaseModel):
    """
    Explainable contributions to the combined score S(h).
    (Must match belief.py ScoreBreakdown fields.)
    """
    retrieval: float = 0.0
    signature_match: float = 0.0
    applicability: float = 0.0
    slot_alignment: float = 0.0
    total: float = 0.0


class BeliefCandidate(BaseModel):
    """
    One hypothesis h (procedure node) in the belief distribution.
    (Must match belief.py BeliefCandidate fields.)
    """
    procedure_id: str
    prob: float = Field(..., ge=0.0, le=1.0, description="Softmax probability over candidates")
    score: float = Field(..., description="Raw combined score before softmax")
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)

    missing_slots: List[str] = Field(default_factory=list, description="Slots needed for safe execution")
    conflicts: List[str] = Field(default_factory=list, description="Precondition contradictions")
    risk: RiskLevel = "low"
    signature_strength: SigStrength = "none"


class Belief(BaseModel):
    """
    Belief state over procedures (hypotheses).
    """
    procedures: List[BeliefCandidate] = Field(
        default_factory=list,
        description="Sorted by prob desc (top hypothesis first)"
    )


# -----------------------------
# 3) Uncertainty (matches belief.py)
# -----------------------------

class Uncertainty(BaseModel):
    """
    Uncertainty metrics derived from belief distribution and slot coverage.
    (Must match belief.py UncertaintyMetrics fields.)
    """
    entropy: float = Field(0.0, ge=0.0, description="Entropy of belief distribution")
    entropy_norm: float = Field(0.0, ge=0.0, le=1.0, description="Entropy normalized by log(K)")

    top1: float = Field(0.0, ge=0.0, le=1.0, description="Highest procedure probability")
    top2: float = Field(0.0, ge=0.0, le=1.0, description="Second highest procedure probability")
    margin: float = Field(0.0, ge=0.0, le=1.0, description="top1 - top2")

    coverage_top1: float = Field(
        0.0, ge=0.0, le=1.0,
        description="% of required slots filled for top hypothesis"
    )

    conflicts_top1: int = Field(0, ge=0, description="Number of contradictions for top hypothesis")
    signature_strength_top1: SigStrength = Field("none", description="Signature strength for top hypothesis")


# -----------------------------
# Full State
# -----------------------------

ActionType = Literal["ASK", "ACT", "VERIFY", "REQUEST_ARTIFACT", "ESCALATE", "NOOP"]


class Action(BaseModel):
    type: ActionType
    payload: Dict[str, Any] = Field(default_factory=dict)


class State(BaseModel):
    """
    Single source of truth for the controller.
    """
    conversation_id: str
    turn_id: int = 0

    evidence: Evidence
    belief: Belief = Field(default_factory=Belief)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)

    last_action: Optional[Action] = None
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Optional: compact event log for debugging; keep small in prod"
    )


# -----------------------------
# Convenience: ExtractionBundle -> frame dict (for belief.py)
# -----------------------------
# belief.py expects: Frame = Dict[str, SlotValue(value, conf, source)]
# We provide a compatible SlotValue dataclass and a canonicalizing converter.

@dataclass
class SlotValue:
    value: Optional[str] = None
    conf: SlotConf = "unknown"
    source: Optional[str] = None


def _canon_key(k: str) -> str:
    return (k or "").strip().lower()


def extraction_to_frame(
    extractions: ExtractionBundle,
    *,
    include_entities_as_slots: bool = False,
) -> Dict[str, SlotValue]:
    """
    Convert ExtractionBundle into the Frame mapping expected by belief.build_belief().

    - Keys are canonicalized to lowercase/strip to match belief's normalized lookup.
    - slots become frame[slot_name]
    - signatures are also inserted (e.g., error_signature/error_code/error_message)
    - entities can optionally be merged as slots if you want them accessible for slot_alignment
      (default False to avoid polluting slot namespace).
    """
    frame: Dict[str, SlotValue] = {}

    for f in extractions.slots:
        key = _canon_key(f.name)
        if not key:
            continue
        frame[key] = SlotValue(value=f.value, conf=f.conf, source=f.source)

    for s in extractions.signatures:
        key = _canon_key(s.name)
        if not key:
            continue
        frame[key] = SlotValue(value=s.value, conf=s.conf, source=s.source)

    if include_entities_as_slots:
        for e in extractions.entities:
            key = _canon_key(e.name)
            if not key:
                continue
            if key not in frame:
                frame[key] = SlotValue(value=e.value, conf=e.conf, source=e.source)

    return frame


# -----------------------------
# Demo
# -----------------------------

if __name__ == "__main__":
    st = State(
        conversation_id="conv-001",
        turn_id=1,
        evidence=Evidence(
            user_signal=UserSignal(type="chat", text="My ThinkCell tab is missing in PowerPoint."),
            extractions=ExtractionBundle(
                slots=[
                    ExtractedField(name="product", value="ThinkCell", conf="high", source="llm"),
                    ExtractedField(name="application", value="PowerPoint", conf="high", source="llm"),
                    ExtractedField(name="os", value="windows", conf="medium", source="llm"),
                ],
                signatures=[
                    ExtractedField(name="error_signature", value="ThinkCell tab missing", conf="medium", source="llm")
                ],
                entities=[],
            ),
            retrieval_candidates=[
                RetrievalCandidate(
                    procedure_id="proc:abc",
                    retrieval_score=2.1,
                    title="Basic troubleshooting for ThinkCell plugin issues",
                    risk="low",
                ),
                RetrievalCandidate(
                    procedure_id="proc:def",
                    retrieval_score=1.7,
                    title="Install ThinkCell from corporate package",
                    risk="medium",
                ),
            ],
        ),
        last_action=Action(type="NOOP", payload={}),
    )

    print(st.model_dump_json(indent=2))
    print("Frame:", extraction_to_frame(st.evidence.extractions))
