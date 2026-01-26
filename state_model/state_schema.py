#!/usr/bin/env python3
"""
state_schema.py

State schema for an IT-support uncertainty-to-certainty agent.

Includes:
1) Evidence
   - user signal (chat/voice)
   - extractions from signal (slots + confidence, signatures, entities)
2) Belief Object
   - procedure hypotheses with probability + explainable score breakdown
3) Uncertainty
   - entropy
   - top1/top2/margin
   - coverage for top hypothesis

This is intentionally "controller-friendly":
- LLM modules fill Evidence.extractions
- Retriever fills Evidence.retrieval_candidates
- Belief module fills Belief + Uncertainty deterministically

Dependencies:
  pip install pydantic
"""

from __future__ import annotations

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
    # Optional metadata you may want later
    language: Optional[str] = Field(None, description="BCP-47 tag if available, e.g., en-US")
    timestamp_ms: Optional[int] = Field(None, description="Client timestamp in ms")
    # If voice is used, attach the source as needed
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
    This is the evidence-to-structure bridge.
    """
    # Canonical slots you care about; keep flexible as list/dict.
    slots: List[ExtractedField] = Field(default_factory=list, description="Slot-like extracted fields")
    # Error messages/codes/signatures (also ExtractedField)
    signatures: List[ExtractedField] = Field(default_factory=list, description="Error signatures/codes/messages")
    # Entities (product/app/plugin names, etc.)
    entities: List[ExtractedField] = Field(default_factory=list, description="Named entities relevant to IT support")
    # Any freeform notes from extractor (optional)
    notes: List[str] = Field(default_factory=list, description="Optional extraction notes or ambiguities")


class RetrievalCandidate(BaseModel):
    """
    Output from your retriever for this turn before belief scoring.
    """
    procedure_id: str
    retrieval_score: float
    doc_id: Optional[str] = None
    doc_title: Optional[str] = None
    title: Optional[str] = None


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
# 2) Belief object
# -----------------------------

RiskLevel = Literal["low", "medium", "high"]
SigStrength = Literal["none", "weak", "regex", "exact"]


class ScoreBreakdown(BaseModel):
    """
    Explainable contributions to the combined score S(h).
    """
    retrieval: float = 0.0
    signature_match: float = 0.0
    applicability: float = 0.0
    slot_alignment: float = 0.0
    total: float = 0.0


class BeliefCandidate(BaseModel):
    """
    One hypothesis h (procedure node) in the belief distribution.
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
# 3) Uncertainty
# -----------------------------

class Uncertainty(BaseModel):
    """
    Uncertainty metrics derived from belief distribution and slot coverage.
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

    # Optional but often helpful
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
# Convenience: Convert ExtractionBundle -> frame dict (for belief.py)
# -----------------------------
# belief.py uses: Frame = Dict[str, SlotValue(value, conf, source)]
# We'll provide a helper that creates that mapping.

from dataclasses import dataclass

@dataclass
class SlotValue:
    value: Optional[str] = None
    conf: SlotConf = "unknown"
    source: Optional[str] = None


def extraction_to_frame(extractions: ExtractionBundle) -> Dict[str, SlotValue]:
    """
    Convert your ExtractionBundle into a frame dict expected by build_belief().

    - slots become frame[slot_name]
    - signatures are also inserted (e.g., error_signature/error_code/error_message)
    - entities can optionally be merged as slots (conservative: not by default)
    """
    frame: Dict[str, SlotValue] = {}

    for f in extractions.slots:
        frame[f.name] = SlotValue(value=f.value, conf=f.conf, source=f.source)

    for s in extractions.signatures:
        frame[s.name] = SlotValue(value=s.value, conf=s.conf, source=s.source)

    # If you want entities accessible as slots, you can enable this:
    # for e in extractions.entities:
    #     if e.name not in frame:
    #         frame[e.name] = SlotValue(value=e.value, conf=e.conf, source=e.source)

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
                RetrievalCandidate(procedure_id="proc:abc", retrieval_score=2.1, title="Basic troubleshooting for ThinkCell plugin issues"),
                RetrievalCandidate(procedure_id="proc:def", retrieval_score=1.7, title="Install ThinkCell from corporate package"),
            ]
        ),
        last_action=Action(type="NOOP", payload={}),
    )

    print(st.model_dump_json(indent=2))
    print("Frame:", extraction_to_frame(st.evidence.extractions))
