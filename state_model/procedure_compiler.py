#!/usr/bin/env python3
"""
procedure_compiler.py

LLM-powered ProcedureNode/Lane builder (KB doc -> procedure_nodes)

This compiler now emits *conversation-grounded* procedures suitable for an agent that can only observe
through user chat/voice + user-provided artifacts.

- Facts-first schema (FactSpec) to stabilize extraction + belief updates.
- Step-level EvidenceActions (what to ask / request after each step).
- Step-level ValidationRules and branching transitions driven by extracted facts.
- Node remains node-driven (ask_slots, slot_questions, slot_priority, slot_safety, artifact_requests),
  but now also produces structured evidence/validation contracts inside steps.

Important design principle:
- LLM proposes structured outputs; your runtime can enforce budgets/hysteresis/graph validity.
- All conditions reference ONLY fact.* / slot.* / artifact.* / ctx.* / belief.* (no "agent observed UI").

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv()

logger = logging.getLogger("procedure_compiler")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

# =============================================================================
# Types / enums
# =============================================================================

RiskLevel = Literal["low", "medium", "high"]

# We keep StepKind close to runtime loop. In user-only environments, "command/ui" are guidance, not executable.
StepKind = Literal["instruction", "check", "decision", "artifact", "escalation", "rollback", "note"]

SlotSafety = Literal["safe", "sensitive", "blocked"]

ArtifactType = Literal["text", "screenshot", "logs", "file", "recording", "link", "other"]

EvidenceActionKind = Literal["question", "artifact_request", "confirmation"]

FactType = Literal["string", "boolean", "enum", "number", "list_string", "json"]

Comparator = Literal[
    "eq",
    "neq",
    "in",
    "nin",
    "contains",
    "regex",
    "exists",
    "not_exists",
    "gte",
    "lte",
]

ValidationEffect = Literal["increase_belief", "decrease_belief", "terminal_resolved", "terminal_escalate"]


# =============================================================================
# Models (NEW)
# =============================================================================

_ALLOWED_LHS_PREFIX = ("fact.", "slot.", "artifact.", "ctx.", "belief.")


class Condition(BaseModel):
    """
    Condition evaluated at runtime over extracted facts / slots / artifact flags / context / belief signals.
    lhs must start with one of:
      fact.<key> | slot.<key> | artifact.<key> | ctx.<key> | belief.<key>
    """
    lhs: str
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
    """
    all_of must all hold; any_of at least one (if provided); none_of must not hold.
    """
    all_of: List[Condition] = Field(default_factory=list)
    any_of: List[Condition] = Field(default_factory=list)
    none_of: List[Condition] = Field(default_factory=list)


class FactSpec(BaseModel):
    """
    Stable fact taxonomy entry. This is the contract between:
      - LLM extractor (turn -> facts)
      - belief updater / validators (facts -> confidence updates)
      - evidence actions (questions request specific facts)
    """
    fact: str = Field(..., description="Stable fact key, e.g. 'os', 'app', 'error_code', 'vpn_on'")
    description: str
    type: FactType = "string"
    allowed_values: List[str] = Field(default_factory=list)
    extraction_hint: Optional[str] = None
    required_for_certainty: bool = False

    @field_validator("fact")
    @classmethod
    def _fact_key(cls, v: str) -> str:
        v = (v or "").strip()
        if not re.match(r"^[a-zA-Z0-9_\-:.]+$", v):
            raise ValueError("fact must be url-safe-ish (letters/digits/_-:.)")
        return v.lower()


class EvidenceRequest(BaseModel):
    type: ArtifactType = Field("text", description="What to request from the user")
    prompt: str = Field(..., min_length=1, description="User-facing prompt")
    targets: List[str] = Field(default_factory=list, description="Fact keys expected to extract from response")
    examples: List[str] = Field(default_factory=list, description="Example user responses (optional)")
    redact_hints: List[str] = Field(default_factory=list, description="PII hints: ['email','phone','token']")
    parse_hint: Optional[str] = Field(None, description="How to parse/what to extract (optional)")

    @field_validator("targets")
    @classmethod
    def _targets_keys(cls, v: List[str]) -> List[str]:
        for k in v or []:
            if not re.match(r"^[a-zA-Z0-9_\-:.]+$", k):
                raise ValueError(f"Invalid target fact key: {k}")
        return [x.lower() for x in (v or [])]


class EvidenceAction(BaseModel):
    """
    This is one of your "uncertainty circles":
    A specific question or artifact request whose result yields facts.
    """
    action_id: str = Field(..., description="Stable ID for this evidence action")
    kind: EvidenceActionKind = Field("question", description="question/artifact_request/confirmation")
    intent: str = Field(..., description="Why we ask; what it disambiguates")
    request: EvidenceRequest

    # Selection metadata (optional but useful)
    cost: Literal["low", "med", "high"] = "low"
    expected_info_gain: float = Field(0.0, ge=0.0, le=1.0, description="Heuristic; optional")

    @field_validator("action_id")
    @classmethod
    def _id_safe(cls, v: str) -> str:
        v = (v or "").strip()
        if not re.match(r"^[a-zA-Z0-9_\-:.]+$", v):
            raise ValueError("action_id must be url-safe-ish")
        return v


class ValidationRule(BaseModel):
    """
    Used by runtime to update belief and/or stop early based on extracted facts.
    """
    rule_id: str
    description: str
    when: ConstraintSet = Field(default_factory=ConstraintSet)
    effect: ValidationEffect = "increase_belief"
    weight: float = Field(1.0, ge=0.0, le=10.0)

    @field_validator("rule_id")
    @classmethod
    def _rid_safe(cls, v: str) -> str:
        v = (v or "").strip()
        if not re.match(r"^[a-zA-Z0-9_\-:.]+$", v):
            raise ValueError("rule_id must be url-safe-ish")
        return v


class TransitionEdge(BaseModel):
    """
    Branching transitions between steps (graph-like inside a lane).
    Runtime may evaluate when->next.
    """
    to_step_id: str
    when: ConstraintSet = Field(default_factory=ConstraintSet)
    rationale: Optional[str] = None


class ArtifactRequest(BaseModel):
    # procedure-level artifact request (from compiled lanes)
    # keep tolerant to compiler variations
    type: Literal["screenshot", "text", "logs", "file", "audio", "recording"] = "text"
    prompt: str
    when: str = ""
    redact_hints: List[str] = Field(default_factory=list)

    # compiler may emit null; accept it
    parse_hint: Optional[str] = None

    @field_validator("when", mode="before")
    @classmethod
    def _coerce_when(cls, v: Any) -> str:
        """Coerce None to empty string for when field."""
        if v is None:
            return ""
        if isinstance(v, str):
            return v.strip()
        return str(v)

    @field_validator("parse_hint", mode="before")
    @classmethod
    def _coerce_parse_hint(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return str(v)



class Precondition(BaseModel):
    slot: str = Field(..., description="Slot name, e.g. 'os', 'product', 'version'")
    values: List[str] = Field(default_factory=list, description="Allowed values for the slot")

    @field_validator("slot")
    @classmethod
    def _validate_slot(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-:.]+$", v.strip()):
            raise ValueError("precondition.slot must be url-safe-ish")
        return v.strip().lower()


class ProcedureStep(BaseModel):
    """
    Step is now "instruction + evidence + validation + branching".
    NOTE: We keep expected_observation as optional legacy/human readability field.
    """
    step_id: str = Field(..., description="Stable ID within this procedure, like 's1', 's2'")
    kind: StepKind = Field("instruction")
    title: Optional[str] = Field(None, description="Short step title (optional)")
    instruction: str = Field(..., min_length=1, description="What the user should do")

    # guidance only (not agent-executed)
    how_to_reach: List[str] = Field(default_factory=list, description="UI path / user commands as bullets")

    # evidence contract
    evidence_actions: List[EvidenceAction] = Field(default_factory=list, description="What to ask/request after this step")

    # belief/termination signals
    validations: List[ValidationRule] = Field(default_factory=list)

    # pass/fail and branching transitions (optional)
    pass_when: ConstraintSet = Field(default_factory=ConstraintSet)
    fail_when: ConstraintSet = Field(default_factory=ConstraintSet)
    on_pass: List[TransitionEdge] = Field(default_factory=list)
    on_fail: List[TransitionEdge] = Field(default_factory=list)

    # ops/safety
    requires_admin: bool = Field(False)
    optional: bool = Field(False)
    safety_notes: Optional[str] = None
    expected_observation: Optional[str] = Field(
        None,
        description="Legacy human-readable observation. Do NOT rely on this operationally.",
    )

    @field_validator("step_id")
    @classmethod
    def _validate_step_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("step_id must be alphanumeric/underscore/dash")
        return v

class SlotQuestion(BaseModel):
    slot: str
    question: str

class SlotSafety(BaseModel):
    slot: str
    safety: Literal["safe", "sensitive", "blocked"]

class ProcedureNode(BaseModel):
    """
    This is effectively your "ProcedureLane".
    We keep the name ProcedureNode for compatibility with your pipeline, but semantics are lane-like.
    """
    procedure_id: str = Field(..., description="Stable ID (computed deterministically)")
    doc_id: str = Field(...)
    doc_title: str = Field(...)
    title: str = Field(...)
    summary: str = Field(..., description="1-3 sentence summary of when to use this procedure")

    preconditions: List[Precondition] = Field(default_factory=list)

    # Facts taxonomy relevant for this node (drives extraction and evidence/validation)
    fact_specs: List[FactSpec] = Field(default_factory=list)

    # Gating inputs needed to ACT safely
    required_slots: List[str] = Field(default_factory=list)

    # Retrieval
    signatures: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    # Steps (now evidence+validation aware)
    entry_step_id: str = Field("s1", description="First step to execute")
    steps: List[ProcedureStep] = Field(..., min_length=1)

    # Verifications / escalation
    verify_prompts: List[str] = Field(default_factory=list)
    escalate_requirements: List[str] = Field(default_factory=list)
    risk: RiskLevel = Field("low")

    # Node-driven ask plan / safety
    ask_slots: List[str] = Field(default_factory=list)
    slot_priority: List[str] = Field(default_factory=list)
    slot_questions: List[SlotQuestion] = Field(default_factory=list)
    slot_safety: List[SlotSafety] = Field(default_factory=list)

    # Fallback artifacts / uncertainty questions
    artifact_requests: List[ArtifactRequest] = Field(default_factory=list)
    ask_if_uncertain: List[str] = Field(default_factory=list)

    # Hybrid retrieval indices (added for enhanced retrieval)
    fact_keywords: List[str] = Field(default_factory=list, description="BM25 keywords for retrieval")
    fact_embeddings: List[List[float]] = Field(default_factory=list, description="Dense embeddings for semantic search")
    fact_descriptions: List[str] = Field(default_factory=list, description="Cross-encoder descriptions for reranking")

    @field_validator("procedure_id")
    @classmethod
    def _validate_procedure_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-:.]+$", v):
            raise ValueError("procedure_id must be URL/filename-safe")
        return v


class ProcedureNodesResponse(BaseModel):
    procedure_nodes: List[ProcedureNode] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# =============================================================================
# JSON Schema for Structured Outputs (UPDATED)
# =============================================================================
def procedure_nodes_json_schema() -> Dict[str, Any]:
    """
    We require:
    - ask plan fields (node-driven)
    - step evidence actions + validations + branching
    - fact_specs for stable extraction keys
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "procedure_nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "procedure_id": {"type": "string"},
                        "doc_id": {"type": "string"},
                        "doc_title": {"type": "string"},
                        "title": {"type": "string"},
                        "summary": {"type": "string"},

                        "preconditions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "slot": {"type": "string"},
                                    "values": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["slot", "values"],
                            },
                        },

                        "fact_specs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "fact": {"type": "string"},
                                    "description": {"type": "string"},
                                    "type": {"type": "string", "enum": ["string", "boolean", "enum", "number", "list_string", "json"]},
                                    "allowed_values": {"type": "array", "items": {"type": "string"}},
                                    "extraction_hint": {"type": ["string", "null"]},
                                    "required_for_certainty": {"type": "boolean"},
                                },
                                "required": ["fact", "description", "type", "allowed_values", "extraction_hint", "required_for_certainty"],
                            },
                        },

                        "required_slots": {"type": "array", "items": {"type": "string"}},
                        "signatures": {"type": "array", "items": {"type": "string"}},
                        "tags": {"type": "array", "items": {"type": "string"}},

                        "entry_step_id": {"type": "string"},

                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "step_id": {"type": "string"},
                                    "kind": {"type": "string", "enum": ["instruction", "check", "decision", "artifact", "escalation", "rollback", "note"]},
                                    "title": {"type": ["string", "null"]},
                                    "instruction": {"type": "string"},
                                    "how_to_reach": {"type": "array", "items": {"type": "string"}},

                                    "evidence_actions": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "action_id": {"type": "string"},
                                                "kind": {"type": "string", "enum": ["question", "artifact_request", "confirmation"]},
                                                "intent": {"type": "string"},
                                                "cost": {"type": "string", "enum": ["low", "med", "high"]},
                                                "expected_info_gain": {"type": "number"},
                                                "request": {
                                                    "type": "object",
                                                    "additionalProperties": False,
                                                    "properties": {
                                                        "type": {"type": "string", "enum": ["text", "screenshot", "logs", "file", "recording", "link", "other"]},
                                                        "prompt": {"type": "string"},
                                                        "targets": {"type": "array", "items": {"type": "string"}},
                                                        "examples": {"type": "array", "items": {"type": "string"}},
                                                        "redact_hints": {"type": "array", "items": {"type": "string"}},
                                                        "parse_hint": {"type": ["string", "null"]},
                                                    },
                                                    "required": ["type", "prompt", "targets", "examples", "redact_hints", "parse_hint"],
                                                },
                                            },
                                            "required": ["action_id", "kind", "intent", "request", "cost", "expected_info_gain"],
                                        },
                                    },

                                    "validations": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "rule_id": {"type": "string"},
                                                "description": {"type": "string"},
                                                "effect": {"type": "string", "enum": ["increase_belief", "decrease_belief", "terminal_resolved", "terminal_escalate"]},
                                                "weight": {"type": "number"},
                                                "when": {
                                                    "type": "object",
                                                    "additionalProperties": False,
                                                    "properties": {
                                                        "all_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                                        "any_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                                        "none_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                                    },
                                                    "required": ["all_of", "any_of", "none_of"],
                                                },
                                            },
                                            "required": ["rule_id", "description", "when", "effect", "weight"],
                                        },
                                    },

                                    "pass_when": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "all_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                            "any_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                            "none_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                        },
                                        "required": ["all_of", "any_of", "none_of"],
                                    },
                                    "fail_when": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "all_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                            "any_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                            "none_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                        },
                                        "required": ["all_of", "any_of", "none_of"],
                                    },

                                    "on_pass": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "to_step_id": {"type": "string"},
                                                "rationale": {"type": ["string", "null"]},
                                                "when": {
                                                    "type": "object",
                                                    "additionalProperties": False,
                                                    "properties": {
                                                        "all_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                                        "any_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                                        "none_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                                    },
                                                    "required": ["all_of", "any_of", "none_of"],
                                                },
                                            },
                                            "required": ["to_step_id", "when", "rationale"],
                                        },
                                    },
                                    "on_fail": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "to_step_id": {"type": "string"},
                                                "rationale": {"type": ["string", "null"]},
                                                "when": {
                                                    "type": "object",
                                                    "additionalProperties": False,
                                                    "properties": {
                                                        "all_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                                        "any_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                                        "none_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                                    },
                                                    "required": ["all_of", "any_of", "none_of"],
                                                },
                                            },
                                            "required": ["to_step_id", "when", "rationale"],
                                        },
                                    },

                                    "requires_admin": {"type": "boolean"},
                                    "optional": {"type": "boolean"},
                                    "safety_notes": {"type": ["string", "null"]},
                                    "expected_observation": {"type": ["string", "null"]},
                                },
                                "required": [
                                    "step_id",
                                    "kind",
                                    "title",
                                    "instruction",
                                    "how_to_reach",
                                    "evidence_actions",
                                    "validations",
                                    "pass_when",
                                    "fail_when",
                                    "on_pass",
                                    "on_fail",
                                    "requires_admin",
                                    "optional",
                                    "safety_notes",
                                    "expected_observation",
                                ],
                            },
                        },

                        "verify_prompts": {"type": "array", "items": {"type": "string"}},
                        "escalate_requirements": {"type": "array", "items": {"type": "string"}},
                        "risk": {"type": "string", "enum": ["low", "medium", "high"]},

                        "ask_slots": {"type": "array", "items": {"type": "string"}},
                        "slot_priority": {"type": "array", "items": {"type": "string"}},
                        "slot_questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "slot": {"type": "string"},
                                    "question": {"type": "string"},
                                },
                                "required": ["slot", "question"],
                            },
                        },
                        "slot_safety": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "slot": {"type": "string"},
                                    "safety": {"type": "string", "enum": ["safe", "sensitive", "blocked"]},
                                },
                                "required": ["slot", "safety"],
                            },
                        },
                        "artifact_requests": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "type": {"type": "string", "enum": ["text", "screenshot", "logs", "file", "recording", "link", "other"]},
                                    "prompt": {"type": "string"},
                                    "when": {"type": ["string", "null"]},
                                    "redact_hints": {"type": "array", "items": {"type": "string"}},
                                    "parse_hint": {"type": ["string", "null"]},
                                },
                                "required": ["type", "prompt", "when", "redact_hints", "parse_hint"],
                            },
                        },
                        "ask_if_uncertain": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "procedure_id",
                        "doc_id",
                        "doc_title",
                        "title",
                        "summary",
                        "preconditions",
                        "fact_specs",
                        "required_slots",
                        "signatures",
                        "tags",
                        "entry_step_id",
                        "steps",
                        "verify_prompts",
                        "escalate_requirements",
                        "risk",
                        "ask_slots",
                        "slot_priority",
                        "slot_questions",
                        "slot_safety",
                        "artifact_requests",
                        "ask_if_uncertain",
                    ],
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "$defs": {
            "any_json": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                    {
                        "type": "array",
                        "items": {"$ref": "#/$defs/any_json"},
                    },
                    {
                        "type": "object",
                        "additionalProperties": {"$ref": "#/$defs/any_json"},
                    },
                ]
            },
            "condition": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "lhs": {"type": "string"},
                    "op": {
                        "type": "string",
                        "enum": ["eq", "neq", "in", "nin", "contains", "regex", "exists", "not_exists", "gte", "lte"],
                    },
                    "rhs": {"$ref": "#/$defs/any_json"},
                },
                "required": ["lhs", "op", "rhs"],
            },
        },

        "required": ["procedure_nodes", "warnings"],
    }

def procedure_nodes_core_json_schema() -> Dict[str, Any]:
    """
    Pass A: small, stable node schema.
    No validations/branching/evidence_actions in steps.
    """
    base = procedure_nodes_json_schema()

    # Clone and then strip heavy step fields
    import copy
    schema = copy.deepcopy(base)

    # steps schema lives at:
    # properties.procedure_nodes.items.properties.steps.items.properties
    step_props = schema["properties"]["procedure_nodes"]["items"]["properties"]["steps"]["items"]["properties"]
    step_required = schema["properties"]["procedure_nodes"]["items"]["properties"]["steps"]["items"]["required"]

    # Remove heavy fields in Pass A
    for k in ["evidence_actions", "validations", "pass_when", "fail_when", "on_pass", "on_fail"]:
        step_props.pop(k, None)
        if k in step_required:
            step_required.remove(k)

    return schema


def procedure_nodes_step_patch_schema() -> Dict[str, Any]:
    """
    Pass B: enrich steps via patches to avoid huge single-shot JSON.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "procedure_id": {"type": "string"},
            "step_patches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "step_id": {"type": "string"},

                        "evidence_actions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "action_id": {"type": "string"},
                                    "kind": {"type": "string", "enum": ["question", "artifact_request", "confirmation"]},
                                    "intent": {"type": "string"},
                                    "cost": {"type": "string", "enum": ["low", "med", "high"]},
                                    "expected_info_gain": {"type": "number"},
                                    "request": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "type": {"type": "string", "enum": ["text", "screenshot", "logs", "file", "recording", "link", "other"]},
                                            "prompt": {"type": "string"},
                                            "targets": {"type": "array", "items": {"type": "string"}},
                                            "examples": {"type": "array", "items": {"type": "string"}},
                                            "redact_hints": {"type": "array", "items": {"type": "string"}},
                                            "parse_hint": {"type": ["string", "null"]},
                                        },
                                        "required": ["type", "prompt", "targets", "examples", "redact_hints", "parse_hint"],
                                    },
                                },
                                "required": ["action_id", "kind", "intent", "request", "cost", "expected_info_gain"],
                            },
                        },

                        "validations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "rule_id": {"type": "string"},
                                    "description": {"type": "string"},
                                    "effect": {"type": "string", "enum": ["increase_belief", "decrease_belief", "terminal_resolved", "terminal_escalate"]},
                                    "weight": {"type": "number"},
                                    "when": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "all_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                            "any_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                            "none_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                        },
                                        "required": ["all_of", "any_of", "none_of"],
                                    },
                                },
                                "required": ["rule_id", "description", "when", "effect", "weight"],
                            },
                        },

                        "pass_when": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "all_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                "any_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                "none_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                            },
                            "required": ["all_of", "any_of", "none_of"],
                        },
                        "fail_when": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "all_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                "any_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                "none_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                            },
                            "required": ["all_of", "any_of", "none_of"],
                        },

                        "on_pass": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "to_step_id": {"type": "string"},
                                    "rationale": {"type": ["string", "null"]},
                                    "when": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "all_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                            "any_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                            "none_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                        },
                                        "required": ["all_of", "any_of", "none_of"],
                                    },
                                },
                                "required": ["to_step_id", "when", "rationale"],
                            },
                        },
                        "on_fail": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "to_step_id": {"type": "string"},
                                    "rationale": {"type": ["string", "null"]},
                                    "when": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "all_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                            "any_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                            "none_of": {"type": "array", "items": {"$ref": "#/$defs/condition"}},
                                        },
                                        "required": ["all_of", "any_of", "none_of"],
                                    },
                                },
                                "required": ["to_step_id", "when", "rationale"],
                            },
                        },
                    },
                    "required": ["step_id", "evidence_actions", "validations", "pass_when", "fail_when", "on_pass", "on_fail"],
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "$defs": procedure_nodes_json_schema()["$defs"],
        "required": ["procedure_id", "step_patches", "warnings"],
    }


# =============================================================================
# Helpers
# =============================================================================

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def stable_procedure_id(doc_id: str, title: str, summary: str) -> str:
    base = f"{doc_id}::{title.strip().lower()}::{summary.strip().lower()[:120]}"
    return f"proc:{sha1_hex(base)[:16]}"


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "procedure"


def extract_output_text(resp: Any) -> str:
    """
    Works with OpenAI Responses API result. Keeps your existing behavior.
    """
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text.strip():
        return resp.output_text
    try:
        parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None)
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def chunk_text(text: str, max_chars: int = 18000) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    headings = [m.start() for m in re.finditer(r"(?m)^(#{1,6}\s+.+)$", text)]
    if len(headings) <= 1:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    idxs = headings + [len(text)]
    chunks: List[str] = []
    cur = ""
    for i in range(len(idxs) - 1):
        seg = text[idxs[i]:idxs[i + 1]]
        if len(cur) + len(seg) <= max_chars:
            cur += seg
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = seg
    if cur.strip():
        chunks.append(cur.strip())

    final: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            final.extend([c[i:i + max_chars] for i in range(0, len(c), max_chars)])
    return final


def _canon_slot(s: str) -> str:
    """
    Canonicalize slot/fact keys to stable keys used by runtime.
    """
    s = (s or "").strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-:.]", "", s)
    return s.lower()


def _uniq(seq: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in seq:
        x2 = (x or "").strip()
        if not x2:
            continue
        k = x2.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x2)
    return out


# =============================================================================
# Slot safety classification (compile-time)
# =============================================================================

_SENSITIVE_SLOT_PATTERNS = [
    r"password",
    r"passcode",
    r"pin\b",
    r"otp\b",
    r"\b2fa\b",
    r"mfa\b",
    r"security_?question",
    r"ssn\b",
    r"social_?security",
    r"credit_?card",
    r"card_?number",
    r"cvv\b",
    r"bank",
    r"routing",
    r"account_?number",
    r"private_?key",
    r"secret\b",
    r"token\b",
    r"api_?key",
]

_BLOCKED_SLOT_PATTERNS = [
    r"company_password",
    r"security_question_answer",
    r"one_time_password",
    r"otp_code",
    r"mfa_code",
    r"2fa_code",
    r"ssn",
    r"credit_card",
    r"card_number",
    r"cvv",
    r"private_key",
    r"secret",
]


def classify_slot_safety(slot: str) -> SlotSafety:
    s = _canon_slot(slot)
    if not s:
        return "blocked"

    for pat in _BLOCKED_SLOT_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return "blocked"

    for pat in _SENSITIVE_SLOT_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return "sensitive"

    return "safe"


# =============================================================================
# Default fact taxonomy mapping (compile-time hardening)
# =============================================================================

# This small mapping makes your facts stable even if the LLM varies.
# You can expand this over time.
_DEFAULT_FACT_SPECS: Dict[str, FactSpec] = {
    "os": FactSpec(fact="os", description="Operating system (windows/macos/linux/ios/android)", type="enum",
                  allowed_values=["windows", "macos", "linux", "ios", "android"], extraction_hint="Normalize to one of allowed values.", required_for_certainty=True),
    "device_type": FactSpec(fact="device_type", description="Device type (laptop/desktop/mobile)", type="enum",
                           allowed_values=["laptop", "desktop", "mobile"], extraction_hint="If user says phone -> mobile.", required_for_certainty=False),
    "app": FactSpec(fact="app", description="Application affected", type="string", extraction_hint="E.g., outlook, chrome, teams.", required_for_certainty=True),
    "product": FactSpec(fact="product", description="Product/tool name", type="string", required_for_certainty=False),
    "plugin": FactSpec(fact="plugin", description="Plugin/add-in name", type="string", required_for_certainty=False),
    "version": FactSpec(fact="version", description="App/OS version", type="string", required_for_certainty=False),
    "error_message": FactSpec(fact="error_message", description="Exact error message text", type="string",
                              extraction_hint="Prefer copy/paste string from user or screenshot OCR.", required_for_certainty=True),
    "error_code": FactSpec(fact="error_code", description="Error code shown (if any)", type="string",
                           extraction_hint="Extract codes like 0x..., HTTP 403, etc.", required_for_certainty=True),
    "vpn_on": FactSpec(fact="vpn_on", description="Whether VPN is connected", type="boolean",
                       extraction_hint="Yes/No -> true/false", required_for_certainty=False),
    "network_location": FactSpec(fact="network_location", description="Network context (office/home/vpn)", type="enum",
                                 allowed_values=["office", "home", "vpn", "unknown"], required_for_certainty=False),
    "scope": FactSpec(fact="scope", description="Whether issue affects only this user or multiple users", type="enum",
                      allowed_values=["single_user", "multiple_users", "unknown"], required_for_certainty=False),
    "user_says_resolved": FactSpec(fact="user_says_resolved", description="User indicates issue is resolved", type="boolean",
                                   required_for_certainty=False),
}


def ensure_fact_specs(node: ProcedureNode) -> None:
    """
    Ensure node.fact_specs exists and is stable.
    Strategy:
    - If LLM provided fact_specs, canonicalize keys and keep them (but normalize types).
    - Always include defaults for common ask/required/precondition slots when recognized.
    """
    # Gather implied keys from node-driven ask plan and required/preconditions
    implied: List[str] = []
    implied.extend([_canon_slot(x) for x in (node.ask_slots or []) if _canon_slot(x)])
    implied.extend([_canon_slot(x) for x in (node.required_slots or []) if _canon_slot(x)])
    implied.extend([_canon_slot(pc.slot) for pc in (node.preconditions or []) if _canon_slot(pc.slot)])

    # Also scan steps for evidence targets
    for st in node.steps or []:
        for ea in st.evidence_actions or []:
            implied.extend([_canon_slot(t) for t in (ea.request.targets or []) if _canon_slot(t)])

    implied = _uniq(implied)

    existing: Dict[str, FactSpec] = {}
    for fs in node.fact_specs or []:
        k = _canon_slot(fs.fact)
        if not k:
            continue
        # normalize
        fs.fact = k
        existing[k] = fs

    # Add known defaults where possible
    for k in implied:
        if k in existing:
            continue
        if k in _DEFAULT_FACT_SPECS:
            existing[k] = _DEFAULT_FACT_SPECS[k]

    # If still empty, add minimal triage set
    if not existing:
        for k in ["os", "app", "error_message", "error_code", "user_says_resolved"]:
            existing[k] = _DEFAULT_FACT_SPECS[k]

    # Ensure stable deterministic order: required_for_certainty first, then alpha
    items = list(existing.values())
    items.sort(key=lambda x: (not bool(x.required_for_certainty), x.fact))
    node.fact_specs = items[:60]


# =============================================================================
# Node-driven ask plan defaults (kept from your prior version)
# =============================================================================

def default_question_for_slot(slot: str) -> str:
    s = _canon_slot(slot)

    if s in ("error_message",):
        return "What is the exact error message (copy/paste if you can)?"
    if s in ("error_code",):
        return "What error code do you see (if any)?"
    if s in ("error_signature",):
        return "What exactly do you see on screen when it happens (the exact message if possible)?"
    if s in ("os",):
        return "Which OS are you on (Windows/macOS/Linux)?"
    if s in ("device_type",):
        return "Is this on a laptop, desktop, or mobile device?"
    if s in ("application", "app"):
        return "Which app is affected?"
    if s in ("product",):
        return "Which product or tool is this related to?"
    if s in ("plugin",):
        return "Is a plugin/add-in involved? If yes, which one?"
    if s in ("version",):
        return "Do you know the app/OS version (or approximate)?"
    if s in ("network_location",):
        return "Are you on office network, home Wi-Fi, or VPN?"
    if s in ("managed_device",):
        return "Is this a company-managed device (MDM/Intune/Jamf) or personal?"
    if s in ("admin_rights",):
        return "Do you have admin rights on this device?"
    if s in ("account_type",):
        return "Is this your company account, personal account, or both?"
    return f"Can you share the {s.replace('_', ' ')}?"


def default_slot_priority(slots: List[str]) -> List[str]:
    prio = [
        "app",
        "application",
        "product",
        "plugin",
        "error_message",
        "error_code",
        "os",
        "device_type",
        "version",
        "network_location",
        "managed_device",
        "admin_rights",
        "account_type",
        "scope",
        "vpn_on",
    ]
    slots_c = [_canon_slot(s) for s in slots if s and _canon_slot(s)]
    sset = set(slots_c)
    out: List[str] = [p for p in prio if p in sset]
    out.extend([s for s in slots_c if s not in set(out)])
    return _uniq(out)


def ensure_node_driven_policy_fields(node: ProcedureNode) -> None:
    required = [_canon_slot(s) for s in (node.required_slots or []) if _canon_slot(s)]
    precond_slots = [_canon_slot(pc.slot) for pc in (node.preconditions or []) if _canon_slot(pc.slot)]
    ask_slots_in = [_canon_slot(x) for x in (node.ask_slots or []) if _canon_slot(x)]
    all_slots = _uniq(required + precond_slots + ask_slots_in)

    # ----------------------------
    # Helpers: normalize list/dict
    # ----------------------------
    def _slot_safety_to_map(v) -> Dict[str, str]:
        if not v:
            return {}
        if isinstance(v, dict):
            out = {}
            for k, val in v.items():
                # guard against tuple-ish keys like ('slot','os')
                if isinstance(k, (tuple, list)) and len(k) == 2 and k[0] == "slot":
                    k = k[1]
                out[str(k)] = str(val)
            return out
        if isinstance(v, list):
            out = {}
            for item in v:
                if item is None:
                    continue
                if isinstance(item, SlotSafety):
                    out[str(item.slot)] = str(item.safety)
                elif isinstance(item, dict):
                    s = str(item.get("slot", "")).strip()
                    sv = str(item.get("safety", "")).strip()
                    if s:
                        out[s] = sv
            return out
        return {}

    def _slot_questions_to_map(v) -> Dict[str, str]:
        if not v:
            return {}
        if isinstance(v, dict):
            out = {}
            for k, val in v.items():
                if isinstance(k, (tuple, list)) and len(k) == 2 and k[0] == "slot":
                    k = k[1]
                out[str(k)] = str(val)
            return out
        if isinstance(v, list):
            out = {}
            for item in v:
                if item is None:
                    continue
                if isinstance(item, SlotQuestion):
                    out[str(item.slot)] = str(item.question)
                elif isinstance(item, dict):
                    s = str(item.get("slot", "")).strip()
                    q = str(item.get("question", "")).strip()
                    if s:
                        out[s] = q
            return out
        return {}

    # ----------------------------
    # slot_safety (normalize -> list)
    # ----------------------------
    safety_map = _slot_safety_to_map(node.slot_safety)

    for s in all_slots:
        if s not in safety_map or not str(safety_map.get(s) or "").strip():
            safety_map[s] = classify_slot_safety(s)

    # clamp to allowed values
    ALLOWED = {"safe", "sensitive", "blocked"}
    for k, v in list(safety_map.items()):
        vv = str(v).strip().lower()
        safety_map[k] = vv if vv in ALLOWED else "safe"

    # store as list[SlotSafety] in deterministic order (all_slots first)
    ordered_safety_slots: List[str] = _uniq(all_slots + [s for s in safety_map.keys() if s not in set(all_slots)])
    node.slot_safety = [SlotSafety(slot=s, safety=safety_map[s]) for s in ordered_safety_slots]

    # hard drop blocked/sensitive from required_slots (compile-time hardening)
    safe_required = [s for s in required if safety_map.get(s, "blocked") == "safe"]
    node.required_slots = _uniq(safe_required)

    # ----------------------------
    # derive ask_slots if empty (safe only)
    # ----------------------------
    ask_slots = [_canon_slot(s) for s in (node.ask_slots or []) if _canon_slot(s)]
    if not ask_slots:
        ask_slots = _uniq(node.required_slots + [s for s in precond_slots if safety_map.get(s, "blocked") == "safe"])

    ask_slots = [s for s in ask_slots if safety_map.get(s, "blocked") == "safe"]
    node.ask_slots = _uniq(ask_slots)

    # ----------------------------
    # slot_questions (normalize -> list)
    # ----------------------------
    sq_map = _slot_questions_to_map(node.slot_questions)

    for s in node.ask_slots:
        if s not in sq_map or not str(sq_map.get(s) or "").strip():
            sq_map[s] = default_question_for_slot(s)
        else:
            q = str(sq_map[s]).strip()
            # normalize to a single trailing '?'
            if q.count("?") >= 2:
                q = q.split("?")[0].strip() + "?"
            elif not q.endswith("?"):
                q = q + "?"
            sq_map[s] = q

    # store as list[SlotQuestion] in deterministic order (ask_slots first)
    ordered_question_slots: List[str] = _uniq(node.ask_slots + [s for s in sq_map.keys() if s not in set(node.ask_slots)])
    node.slot_questions = [SlotQuestion(slot=s, question=sq_map[s]) for s in ordered_question_slots]

    # ----------------------------
    # slot_priority (must be safe + subset of ask_slots)
    # ----------------------------
    sp = [_canon_slot(s) for s in (node.slot_priority or []) if _canon_slot(s)]
    if not sp:
        sp = default_slot_priority(node.ask_slots)

    ask_set = set(node.ask_slots)
    sp2 = [s for s in sp if s in ask_set]
    for s in node.ask_slots:
        if s not in set(sp2):
            sp2.append(s)
    node.slot_priority = _uniq(sp2)

    # ----------------------------
    # fallback artifacts
    # ----------------------------
    if not node.artifact_requests:
        node.artifact_requests = [
            ArtifactRequest(
                type="text",
                prompt="What exactly happens when you try (any error message, or what you see on screen)?",
                when="if issue persists",
                redact_hints=[],
                parse_hint="",
            ),
            ArtifactRequest(
                type="screenshot",
                prompt="If there is an error on screen, please share a screenshot.",
                when="if error is visible",
                redact_hints=[],
                parse_hint="",
            ),
            ArtifactRequest(
                type="logs",
                prompt="If you have logs (or can copy/paste console output), please share the relevant snippet.",
                when="if available",
                redact_hints=["token", "api_key"],
                parse_hint="Extract error lines and timestamps",
            ),
        ]

    # ----------------------------
    # ask_if_uncertain minimal
    # ----------------------------
    if not node.ask_if_uncertain:
        node.ask_if_uncertain = [
            "What are you trying to do, and what is the exact point where it fails (with the exact error text if any)?"
        ]


# =============================================================================
# Postprocessing / normalization
# =============================================================================

def dedupe_procedure_nodes(nodes: List[ProcedureNode]) -> List[ProcedureNode]:
    seen: Dict[str, ProcedureNode] = {}
    for n in nodes:
        sig = "||".join([s.strip().lower() for s in (n.signatures or [])[:6]])
        steps = "||".join([(st.instruction.strip().lower()) for st in (n.steps or [])[:2]])
        ask = "||".join([s.strip().lower() for s in (n.ask_slots or [])[:6]])
        key = sha1_hex(f"{n.doc_id}::{n.title.strip().lower()}::{sig}::{steps}::{ask}")
        if key not in seen:
            seen[key] = n
    return list(seen.values())


def split_multi_action_step(step: ProcedureStep) -> List[ProcedureStep]:
    instr = (step.instruction or "").strip()
    if not instr:
        return [step]
    if len(instr) < 160:
        return [step]

    parts = re.split(r"(?:\.\s+|\n+|;\s+|,\s+then\s+| then\s+)", instr, flags=re.IGNORECASE)
    parts = [p.strip(" -•\t") for p in parts if p and p.strip(" -•\t")]
    if len(parts) < 2 or len(parts) > 7:
        return [step]

    out: List[ProcedureStep] = []
    base_id = slugify(step.step_id)
    for i, p in enumerate(parts, start=1):
        out.append(
            ProcedureStep(
                step_id=f"{base_id}-{i}",
                kind=step.kind,
                title=step.title if i == 1 else None,
                instruction=p if p.endswith(".") else p,
                how_to_reach=step.how_to_reach if i == 1 else [],
                evidence_actions=[],
                validations=[],
                pass_when=ConstraintSet(),
                fail_when=ConstraintSet(),
                on_pass=[],
                on_fail=[],
                requires_admin=step.requires_admin,
                optional=step.optional,
                safety_notes=step.safety_notes,
                expected_observation=None,
            )
        )
    return out


def normalize_preconditions(preconditions: List[Precondition], required_slots: List[str]) -> Tuple[List[Precondition], List[str]]:
    drop_slots = {"issue_status"}
    cleaned: List[Precondition] = []
    for pc in preconditions or []:
        slot = _canon_slot(pc.slot)
        if slot in drop_slots:
            continue
        values = [v.strip() for v in (pc.values or []) if v and v.strip()]
        if not slot:
            continue
        cleaned.append(Precondition(slot=slot, values=values))

    req = [_canon_slot(s) for s in (required_slots or []) if _canon_slot(s) and _canon_slot(s) not in drop_slots]
    req2 = _uniq(req)

    seen_pc = set()
    pc2: List[Precondition] = []
    for pc in cleaned:
        key = (pc.slot, tuple(pc.values))
        if key not in seen_pc:
            seen_pc.add(key)
            pc2.append(pc)

    return pc2, req2


def ensure_soft_signatures(node: ProcedureNode) -> None:
    if node.signatures and any(s.strip() for s in node.signatures):
        return

    t = (node.title or "").lower()
    tags = [x.lower() for x in (node.tags or [])]
    text = " ".join([t] + tags)

    soft: List[str] = []
    if "plugin" in text or "add-in" in text or "addins" in text:
        soft.extend(["plugin not working", "add-in not working", "plugin missing or disabled"])

    proper = re.findall(r"[A-Z][a-zA-Z0-9]+", node.title or "")
    if proper:
        uniq: List[str] = []
        for p in proper:
            if p.lower() not in [u.lower() for u in uniq]:
                uniq.append(p)
            if len(uniq) >= 2:
                break
        for p in uniq:
            soft.append(f"{p} issue")

    if any(k in text for k in ["conference", "meeting", "call", "webex", "teams", "zoom"]):
        soft.extend(["schedule a meeting", "start a meeting", "send meeting invitation"])

    if "vpn" in text or "network" in text:
        soft.extend(["vpn issue", "network connectivity issue"])

    node.signatures = _uniq(soft)[:8]


def ensure_observable_verify_prompts(node: ProcedureNode) -> None:
    v = [x.strip() for x in (node.verify_prompts or []) if x and x.strip()]
    if not v:
        v = ["Did this resolve the issue? (Yes/No)"]
    node.verify_prompts = _uniq(v)


def enrich_tags(node: ProcedureNode) -> None:
    tags = [t.strip() for t in (node.tags or []) if t and t.strip()]
    text = f"{node.title} {node.summary} {' '.join(node.signatures or [])}".lower()

    def add(t: str) -> None:
        tags.append(t)

    if any(k in text for k in ["webex", "cisco webex"]):
        add("webex")
    if "teams" in text:
        add("teams")
    if "zoom" in text:
        add("zoom")
    if any(k in text for k in ["conference", "meeting", "invite", "invitation"]):
        add("meeting")
    if any(k in text for k in ["audio", "mic", "microphone", "speaker"]):
        add("audio")
    if any(k in text for k in ["video", "camera"]):
        add("video")
    if any(k in text for k in ["vpn", "wifi", "ethernet", "network", "proxy", "dns"]):
        add("network")
    if any(k in text for k in ["login", "sign in", "authenticate", "auth"]):
        add("auth")

    node.tags = _uniq(tags)


def normalize_step_contracts(node: ProcedureNode) -> None:
    """
    Compile-time stabilization for evidence targets and condition lhs prefixes.
    - Canonicalize evidence action targets
    - Canonicalize precondition/slot names already done elsewhere
    - Ensure every step has at least one evidence action if step kind implies it
    - Ensure action_id/rule_id uniqueness in a node
    """
    # action_id uniqueness
    seen_action_ids = set()
    seen_rule_ids = set()

    for st in node.steps or []:
        # normalize how_to_reach bullets
        st.how_to_reach = [x.strip() for x in (st.how_to_reach or []) if x and x.strip()]

        # canonicalize evidence targets, dedupe
        for ea in st.evidence_actions or []:
            ea.request.targets = _uniq([_canon_slot(t) for t in (ea.request.targets or []) if _canon_slot(t)])

            # force stable action_id if missing/duplicate
            if not ea.action_id or ea.action_id in seen_action_ids:
                base = slugify(f"{node.procedure_id}-{st.step_id}-{ea.kind}-{ea.intent}")[:32]
                ea.action_id = f"ea:{base}:{sha1_hex(ea.request.prompt)[:6]}"
            seen_action_ids.add(ea.action_id)

        # validations stable ids
        for vr in st.validations or []:
            if not vr.rule_id or vr.rule_id in seen_rule_ids:
                base = slugify(f"{node.procedure_id}-{st.step_id}-{vr.effect}")[:32]
                vr.rule_id = f"vr:{base}:{sha1_hex(vr.description)[:6]}"
            seen_rule_ids.add(vr.rule_id)

        # If step is a "check/decision" and has no evidence action, add minimal confirmation
        if st.kind in ("check", "decision") and not st.evidence_actions:
            st.evidence_actions = [
                EvidenceAction(
                    action_id=f"ea:{slugify(node.procedure_id + '-' + st.step_id)}:confirm",
                    kind="confirmation",
                    intent="Collect the user's outcome for this check",
                    request=EvidenceRequest(
                        type="text",
                        prompt="What happened after you did that? (Include exact error text if any.)",
                        targets=["error_message", "error_code", "user_says_resolved"],
                        examples=[],
                        redact_hints=[],
                        parse_hint="Extract any error text/codes and whether the user says it worked.",
                    ),
                    cost="low",
                    expected_info_gain=0.3,
                )
            ]


def postprocess_node(node: ProcedureNode, doc_id: str, doc_title: str) -> ProcedureNode:
    node.doc_id = doc_id
    node.doc_title = doc_title

    node.procedure_id = stable_procedure_id(doc_id, node.title, node.summary)

    node.preconditions, node.required_slots = normalize_preconditions(node.preconditions, node.required_slots)

    # Ensure entry_step_id exists; default to s1
    if not node.entry_step_id:
        node.entry_step_id = "s1"

    # Split long instructions (keeps step contracts minimal; you can re-add evidence later)
    new_steps: List[ProcedureStep] = []
    for st in node.steps or []:
        new_steps.extend(split_multi_action_step(st))
    node.steps = new_steps or node.steps

    ensure_soft_signatures(node)
    ensure_observable_verify_prompts(node)
    enrich_tags(node)

    node.signatures = [s.strip() for s in (node.signatures or []) if s and s.strip()]
    node.tags = _uniq(node.tags or [])

    ensure_node_driven_policy_fields(node)
    normalize_step_contracts(node)

    # Ensure stable fact taxonomy
    ensure_fact_specs(node)

    # If entry_step_id not found, set to first step id
    step_ids = [s.step_id for s in (node.steps or [])]
    if node.entry_step_id not in step_ids and step_ids:
        node.entry_step_id = step_ids[0]

    return node


# =============================================================================
# LLM Compiler
# =============================================================================

@dataclass
class CompileConfig:
    model: str = "gpt-5.2"
    max_output_tokens: int = 10000
    temperature: float = 0.2
    max_retries: int = 2
    chunk_max_chars: int = 18000


SYSTEM_INSTRUCTIONS = """You are an expert IT support runbook compiler.

Task:
Convert the given knowledge base document into a set of *Procedure Nodes* suitable for an automated troubleshooting agent.

IMPORTANT REALITY CONSTRAINT:
The agent cannot directly observe the user's UI/logs. The agent only learns via:
- user chat/voice responses
- user-provided artifacts (screenshots, copied logs, files)
Therefore you MUST express steps with an evidence contract:
- evidence_actions (questions/artifact requests)
- validations (rules based on extracted facts)
- optional branching transitions (on_pass/on_fail), driven by facts

CRITICAL OUTPUT RULES:
- Output MUST be valid JSON that conforms to the provided JSON Schema.
- No markdown. No extra keys. No extra text.
- Create multiple procedure_nodes if the document contains multiple distinct procedures or variants.

QUALITY RULES:
- Prefer smaller, reusable procedures over one giant node.
- Preconditions: include only if clearly implied (os/product/version/admin/network/etc.).
- Required slots: minimum safe gating slots before acting; DO NOT include secrets/credentials.
- Signatures: extract exact error strings/codes when present; else add 2-8 soft signatures.
- Steps:
  - Each step should be actionable as instructions to the user.
  - Provide "how_to_reach" bullets when the doc has UI navigation or command instructions.
  - Add at least 1 evidence_action per step unless purely informational.
- Evidence actions:
  - Use targets that align with stable IT facts (os, app, version, error_message, error_code, vpn_on, scope, user_says_resolved).
  - Keep prompts short, single-ask.
- Validations:
  - Conditions MUST reference only fact.* (and optionally slot./artifact./ctx./belief. if useful).
  - Use validations to increase/decrease belief and to detect terminal resolve/escalate.
- Branching:
  - Use on_pass/on_fail edges when doc implies a decision or different next step based on user result.

NODE-DRIVEN ASK PLAN (IMPORTANT FOR ORCHESTRATOR):
Your node MUST include fields that let the chat loop ask questions WITHOUT hardcoded allowlists:
- ask_slots: slots safe to ask (non-sensitive). Prefer from required_slots/preconditions.
- slot_priority: ordering of ask_slots (most important first).
- slot_questions: map each ask_slot to EXACTLY ONE short question.
- slot_safety: map relevant slots to safe/sensitive/blocked.
  Never mark credentials/PII secrets as safe (password/pin/otp/2fa/ssn/card/token/secret/api_key).
- artifact_requests: evidence to request if uncertain/unresolved (text error, screenshot, logs, version).
- ask_if_uncertain: 1-3 on-topic questions only.

FACT TAXONOMY:
- Provide fact_specs for the node. Use stable keys where possible (os, device_type, app, product, plugin, version,
  error_message, error_code, vpn_on, network_location, scope, user_says_resolved).
- Do not invent obscure fact keys unless truly necessary.
"""


def compile_procedure_nodes(
    *,
    doc_id: str,
    title: str,
    content: str,
    config: Optional[CompileConfig] = None,
    client: Optional[OpenAI] = None,
) -> List[ProcedureNode]:
    cfg = config or CompileConfig()
    cli = client or OpenAI()

    # Pass A schema (small)
    core_schema = procedure_nodes_core_json_schema()
    # Pass B schema (patches)
    patch_schema = procedure_nodes_step_patch_schema()

    chunks = chunk_text(content, max_chars=cfg.chunk_max_chars)

    all_nodes: List[ProcedureNode] = []
    warnings: List[str] = []

    # -------------------------
    # Helpers
    # -------------------------
    def _call_structured(schema: Dict[str, Any], system: str, user: str, max_out: int) -> Dict[str, Any]:
        last_err: Optional[str] = None
        user_prompt = user

        for attempt in range(cfg.max_retries + 1):
            try:
                resp = cli.responses.create(
                    model=cfg.model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "procedure_nodes",
                            "strict": True,
                            "schema": schema,
                        }
                    },
                    temperature=cfg.temperature,
                    max_output_tokens=max_out,
                )

                raw = extract_output_text(resp)
                if not raw:
                    raise RuntimeError("Empty model output text")

                return json.loads(raw)

            except (json.JSONDecodeError, RuntimeError) as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < cfg.max_retries:
                    time.sleep(0.25 * (attempt + 1))
                    user_prompt = (
                        user_prompt
                        + "\n\nREPAIR INSTRUCTIONS:\n"
                          "- Output ONLY valid JSON matching the schema (no markdown, no extra text).\n"
                          "- Keep strings short and avoid unescaped quotes.\n"
                          "- If unsure, use empty arrays [] and nulls where allowed.\n"
                    )
                    continue
                raise RuntimeError(last_err or "Unknown failure")

        raise RuntimeError(last_err or "Unknown failure")

    def _merge_step_patch(node: Dict[str, Any], patch: Dict[str, Any]) -> None:
        steps = node.get("steps") or []
        by_id = {s.get("step_id"): s for s in steps if isinstance(s, dict) and s.get("step_id")}
        for p in (patch.get("step_patches") or []):
            sid = p.get("step_id")
            if not sid or sid not in by_id:
                continue
            # merge patch fields into step
            for k, v in p.items():
                if k == "step_id":
                    continue
                by_id[sid][k] = v

    def _ensure_step_heavy_fields(node: Dict[str, Any]) -> None:
        """
        After merging, ensure required heavy fields exist on every step,
        even if patch generation skipped them.
        """
        for s in node.get("steps") or []:
            if not isinstance(s, dict):
                continue
            s.setdefault("evidence_actions", [])
            s.setdefault("validations", [])
            s.setdefault("pass_when", {"all_of": [], "any_of": [], "none_of": []})
            s.setdefault("fail_when", {"all_of": [], "any_of": [], "none_of": []})
            s.setdefault("on_pass", [])
            s.setdefault("on_fail", [])

    # -------------------------
    # PASS A: Compile skeleton nodes per chunk
    # -------------------------
    for chunk_i, chunk in enumerate(chunks):
        user_prompt = f"""DOCUMENT METADATA:
            doc_id: {doc_id}
            doc_title: {title}
            chunk: {chunk_i+1}/{len(chunks)}

            DOCUMENT TEXT:
            {chunk}

            INSTRUCTIONS:
            - Produce 1–2 procedure_nodes max.
            - Keep summary <= 240 chars.
            - Keep each step instruction <= 220 chars.
            - steps should be 4–8 max.
            - Do NOT include heavy step fields (evidence_actions/validations/pass_when/fail_when/on_pass/on_fail) in this pass.
            - slot_questions and slot_safety must be ARRAYS of objects: {{slot, question}} / {{slot, safety}}.
            """

        try:
            data = _call_structured(
                core_schema,
                SYSTEM_INSTRUCTIONS,
                user_prompt,
                max_out=min(cfg.max_output_tokens, 2500),
            )
            parsed = ProcedureNodesResponse.model_validate(data)

            for n in parsed.procedure_nodes:
                all_nodes.append(postprocess_node(n, doc_id=doc_id, doc_title=title))
            warnings.extend(parsed.warnings)

        except (ValidationError, RuntimeError) as e:
            warnings.append(f"Failed to compile chunk {chunk_i+1}/{len(chunks)}: {type(e).__name__}: {e}")

    all_nodes = dedupe_procedure_nodes(all_nodes)

    # -------------------------
    # PASS B: Step enrichment per node, in small batches
    # -------------------------
    enriched: List[ProcedureNode] = []
    for node in all_nodes:
        node_dict = node.model_dump()

        steps = node_dict.get("steps") or []
        if not steps:
            enriched.append(node)
            continue

        # Batch steps: 2–3 per call is a good default
        batch_size = getattr(cfg, "patch_batch_size", 2)
        step_ids = [s.get("step_id") for s in steps if isinstance(s, dict) and s.get("step_id")]

        for i in range(0, len(step_ids), batch_size):
            batch = step_ids[i:i + batch_size]

            # Provide only the minimal context needed to patch
            step_summaries = []
            for s in steps:
                if not isinstance(s, dict):
                    continue
                if s.get("step_id") in batch:
                    step_summaries.append({
                        "step_id": s.get("step_id"),
                        "kind": s.get("kind"),
                        "title": s.get("title"),
                        "instruction": s.get("instruction"),
                    })

            patch_prompt = f"""PATCH REQUEST:
procedure_id: {node_dict.get("procedure_id")}
doc_id: {node_dict.get("doc_id")}
doc_title: {node_dict.get("doc_title")}
procedure_title: {node_dict.get("title")}

FACT SPECS (keys you may reference in conditions):
{json.dumps(node_dict.get("fact_specs") or [], ensure_ascii=False)}

STEPS TO PATCH (only these step_ids):
{json.dumps(step_summaries, ensure_ascii=False)}

INSTRUCTIONS:
- Return ONLY JSON matching the patch schema.
- For each step_id above, generate:
  evidence_actions (0-2), validations (0-4),
  pass_when/fail_when (use empty all_of/any_of/none_of when unsure),
  on_pass/on_fail (0-2 transitions each).
- Keep all strings short; avoid long prose.
- Conditions lhs must start with fact./slot./artifact./ctx./belief.
"""

            try:
                patch_data = _call_structured(
                    patch_schema,
                    SYSTEM_INSTRUCTIONS,
                    patch_prompt,
                    max_out=1800,
                )
                # validate shape quickly via pydantic? If you have a PatchResponse model, use it.
                _merge_step_patch(node_dict, patch_data)
                warnings.extend(patch_data.get("warnings") or [])

            except RuntimeError as e:
                warnings.append(f"Failed to patch node={node_dict.get('procedure_id')} steps={batch}: {e}")

        # After patching, ensure required heavy fields exist (even if patch missed a step)
        _ensure_step_heavy_fields(node_dict)

        # Now validate into your ProcedureNode model (this is where you'll catch shape issues)
        try:
            final_node = ProcedureNode.model_validate(node_dict)
            enriched.append(postprocess_node(final_node, doc_id=doc_id, doc_title=title))
        except ValidationError as e:
            warnings.append(f"Failed to validate enriched node={node_dict.get('procedure_id')}: {e}")

    # -------------------------
    # Final postprocess
    # -------------------------
    fixed: List[ProcedureNode] = [postprocess_node(n, doc_id=doc_id, doc_title=title) for n in enriched]
    fixed = dedupe_procedure_nodes(fixed)

    if not fixed and warnings:
        logger.warning(
            "Compilation warnings for doc_id=%s title=%s warnings=%s details=%s",
            doc_id, title, len(warnings), warnings
        )

    return fixed

# =============================================================================
# CSV ingestion + output
# =============================================================================

def _infer_col(headers: List[str], preferred: List[str]) -> Optional[str]:
    hset = {h.strip(): h for h in headers}
    lower_map = {h.lower().strip(): h for h in headers}
    for p in preferred:
        if p in hset:
            return hset[p]
        if p.lower() in lower_map:
            return lower_map[p.lower()]
    return None


def infer_csv_mapping(headers: List[str]) -> Dict[str, str]:
    id_col = _infer_col(headers, ["doc_id", "id", "kb_id", "article_id", "ticket_id", "knowledge_id"])
    title_col = _infer_col(headers, ["title", "doc_title", "subject", "headline", "problem", "issue"])
    content_col = _infer_col(headers, ["content", "text", "body", "article", "knowledge", "resolution", "description"])
    return {"id_col": id_col or "", "title_col": title_col or "", "content_col": content_col or ""}


def compose_content_from_row(row: Dict[str, str], *, primary: str, extras: List[str]) -> str:
    parts: List[str] = []
    if primary and row.get(primary):
        parts.append(str(row.get(primary, "")).strip())

    for c in extras:
        v = str(row.get(c, "")).strip()
        if v and v not in parts:
            parts.append(f"{c}:\n{v}")

    return "\n\n".join([p for p in parts if p])


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def compile_csv_dataset(
    *,
    csv_path: str,
    out_path: str,
    config: Optional[CompileConfig] = None,
    model: Optional[str] = None,
    id_col: Optional[str] = None,
    title_col: Optional[str] = None,
    content_col: Optional[str] = None,
    content_extra_cols: Optional[List[str]] = None,
    start_row: int = 0,
    limit: Optional[int] = None,
    output_format: Literal["jsonl", "json"] = "jsonl",
    progress_path: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = config or CompileConfig()
    if model:
        cfg.model = model

    cli = OpenAI()

    csv_p = Path(csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    progress_p = Path(progress_path) if progress_path else Path(out_path + ".progress.json")
    if progress_p.exists():
        try:
            prog = json.loads(progress_p.read_text(encoding="utf-8"))
            if isinstance(prog, dict) and isinstance(prog.get("next_row"), int) and prog["next_row"] > start_row:
                start_row = int(prog["next_row"])
                logger.info("Resuming from progress file at row=%d", start_row)
        except Exception:
            pass

    compiled_nodes_total = 0
    docs_processed = 0
    docs_failed = 0
    warnings_total: List[str] = []

    json_accum: List[Dict[str, Any]] = []

    if output_format == "jsonl" and start_row == 0 and Path(out_path).exists():
        logger.warning("Overwriting existing output JSONL: %s", out_path)
        Path(out_path).unlink()

    with csv_p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("CSV has no headers/fieldnames")

        headers = list(reader.fieldnames)
        inferred = infer_csv_mapping(headers)

        use_id = id_col or (inferred["id_col"] or None)
        use_title = title_col or (inferred["title_col"] or None)
        use_content = content_col or (inferred["content_col"] or None)

        if not use_title:
            raise RuntimeError(f"Could not infer title column. Pass --title-col. Headers={headers}")
        if not use_content:
            raise RuntimeError(f"Could not infer content column. Pass --content-col. Headers={headers}")

        extra_cols = content_extra_cols or []
        logger.info("CSV mapping: id_col=%s title_col=%s content_col=%s extra_cols=%s", use_id, use_title, use_content, extra_cols)

        for i, row in enumerate(reader):
            if i < start_row:
                continue
            if limit is not None and docs_processed >= limit:
                break

            doc_id = (row.get(use_id, "") if use_id else "").strip()
            if not doc_id:
                doc_id = f"csvrow:{i}"

            title = (row.get(use_title, "") or "").strip() or f"Untitled {doc_id}"
            content = compose_content_from_row(row, primary=use_content, extras=extra_cols)

            if not content.strip():
                docs_failed += 1
                warnings_total.append(f"Row {i}: empty content after composition (doc_id={doc_id})")
                progress_p.write_text(json.dumps({"next_row": i + 1}, indent=2), encoding="utf-8")
                continue

            try:
                nodes = compile_procedure_nodes(
                    doc_id=str(doc_id),
                    title=str(title),
                    content=str(content),
                    config=cfg,
                    client=cli,
                )

                records = []
                for n in nodes:
                    rec = n.model_dump()
                    rec["_source"] = {"csv_path": str(csv_path), "row_index": i}
                    records.append(rec)

                if output_format == "jsonl":
                    write_jsonl(out_path, records)
                else:
                    json_accum.extend(records)

                compiled_nodes_total += len(records)
                docs_processed += 1

                progress_p.write_text(json.dumps({"next_row": i + 1}, indent=2), encoding="utf-8")

                if docs_processed % 10 == 0:
                    logger.info("Processed=%d nodes=%d last_row=%d", docs_processed, compiled_nodes_total, i)

            except Exception as e:
                docs_failed += 1
                msg = f"Row {i} (doc_id={doc_id}) failed: {type(e).__name__}: {e}"
                warnings_total.append(msg)
                logger.exception(msg)
                progress_p.write_text(json.dumps({"next_row": i + 1}, indent=2), encoding="utf-8")
                continue

    if output_format == "json":
        write_json(out_path, json_accum)

    warn_path = out_path + ".warnings.json"
    write_json(warn_path, {"warnings": warnings_total})

    stats = {
        "csv_path": csv_path,
        "out_path": out_path,
        "output_format": output_format,
        "docs_processed": docs_processed,
        "docs_failed": docs_failed,
        "procedure_nodes_written": compiled_nodes_total if output_format == "jsonl" else len(json_accum),
        "warnings_path": warn_path,
        "progress_path": str(progress_p),
        "model": cfg.model,
    }
    return stats


# =============================================================================
# CLI
# =============================================================================

def read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile KB docs into ProcedureNodes (single doc or CSV dataset).")

    # Single doc mode args
    ap.add_argument("--doc-id", type=str, default=None, help="Document ID (single-doc mode)")
    ap.add_argument("--title", type=str, default=None, help="Document title (single-doc mode)")
    ap.add_argument("--content", type=str, default=None, help="Document content (single-doc mode)")
    ap.add_argument("--content-file", type=str, default=None, help="Path to a text/markdown file for content (single-doc mode)")

    # CSV mode args
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV file (dataset mode)")
    ap.add_argument("--id-col", type=str, default=None, help="CSV column name for doc_id (optional)")
    ap.add_argument("--title-col", type=str, default=None, help="CSV column name for title")
    ap.add_argument("--content-col", type=str, default=None, help="CSV column name for primary content/body")
    ap.add_argument("--content-extra-col", action="append", default=[], help="Additional columns to append into content (repeatable)")
    ap.add_argument("--start-row", type=int, default=0, help="Start row index for CSV processing (0-based)")
    ap.add_argument("--limit", type=int, default=None, help="Max number of rows to process")
    ap.add_argument("--output-format", type=str, default="jsonl", choices=["jsonl", "json"], help="Output format (jsonl recommended)")

    # Shared output/model args
    ap.add_argument("--out", type=str, required=True, help="Output path (.jsonl or .json)")
    ap.add_argument("--model", type=str, default=None, help="OpenAI model override (default from CompileConfig)")
    ap.add_argument("--max-output-tokens", type=int, default=2600)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--chunk-max-chars", type=int, default=18000)
    ap.add_argument("--progress-path", type=str, default=None, help="Progress checkpoint file path (CSV mode)")

    args = ap.parse_args()

    cfg = CompileConfig(
        model=args.model or "gpt-4.1",
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        chunk_max_chars=args.chunk_max_chars,
    )

    if args.csv:
        stats = compile_csv_dataset(
            csv_path=args.csv,
            out_path=args.out,
            config=cfg,
            model=args.model,
            id_col=args.id_col,
            title_col=args.title_col,
            content_col=args.content_col,
            content_extra_cols=args.content_extra_col or [],
            start_row=args.start_row,
            limit=args.limit,
            output_format=args.output_format,
            progress_path=args.progress_path,
        )
        print(json.dumps(stats, indent=2))
        return

    if not args.doc_id:
        raise SystemExit("Single-doc mode requires --doc-id (or use --csv).")
    if not args.title:
        raise SystemExit("Single-doc mode requires --title.")
    content = args.content
    if args.content_file:
        content = read_text_file(args.content_file)
    if not content:
        raise SystemExit("Single-doc mode requires --content or --content-file.")

    nodes = compile_procedure_nodes(doc_id=args.doc_id, title=args.title, content=content, config=cfg, client=OpenAI())
    out_records = [n.model_dump() for n in nodes]
    if args.out.endswith(".jsonl"):
        Path(args.out).write_text("", encoding="utf-8")
        write_jsonl(args.out, out_records)
    else:
        write_json(args.out, out_records)
    print(f"Wrote {len(out_records)} procedure nodes to {args.out}")


if __name__ == "__main__":
    main()


"""
Example:

python procedure_compiler.py \
  --csv synthetic_knowledge_items.csv \
  --out ./out/procedure_nodes.jsonl \
  --output-format jsonl \
  --limit 200

python procedure_compiler.py \
  --csv synthetic_knowledge_items.csv \
  --title-col ki_topic \
  --content-col ki_text \
  --out ./out/procedure_nodes.jsonl \
  --output-format jsonl
"""