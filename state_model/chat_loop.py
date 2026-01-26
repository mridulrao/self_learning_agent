#!/usr/bin/env python3
"""
orchestrator.py

User interaction layer + controller that drives the whole loop:

Turn flow:
  1) User sends signal (chat/voice transcript)
  2) LLM extracts slots/signatures/entities -> Evidence.extractions
  3) Retriever returns top-K procedure candidates + retrieval_score
  4) Deterministic belief + uncertainty computed
  5) Deterministic policy chooses next action type: ASK / ACT / VERIFY / REQUEST_ARTIFACT / ESCALATE
  6) LLM composes the user-facing response for that action
  7) Update State, return assistant response + updated state

This file is designed to integrate with your existing modules:
  - procedure_compiler.py produces ProcedureNode objects (stored/indexed elsewhere)
  - belief.py computes belief distribution (we include a compatible build_belief() import)
  - state_schema.py defines State/Evidence/Belief/Uncertainty and extraction_to_frame()

Dependencies:
  pip install openai pydantic python-dotenv

Env:
  OPENAI_API_KEY=...
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# Your schema
from state_schema import (
    Action,
    Evidence,
    ExtractionBundle,
    ExtractedField,
    RetrievalCandidate,
    State,
    Uncertainty,
    UserSignal,
    extraction_to_frame,
    Belief as BeliefObj,
    BeliefCandidate as BeliefObjCandidate,
    ScoreBreakdown as ScoreBreakdownObj,
)

# Your belief computation (the updated one you have)
# from belief import build_belief, SlotValue as BeliefSlotValue
# To make this file standalone-friendly, we’ll import lazily inside functions.

load_dotenv()


# =============================================================================
# 1) LLM: Frame extraction (user signal -> extractions)
# =============================================================================

# --- JSON schema for extraction structured output ---
def extraction_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "slots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": ["string", "null"]},
                        "conf": {"type": "string", "enum": ["unknown", "low", "medium", "high"]},
                        "source": {"type": "string", "enum": ["user", "tool", "llm", "inferred"]},
                        "evidence_snippet": {"type": ["string", "null"]},
                    },
                    "required": ["name", "value", "conf", "source", "evidence_snippet"],
                },
            },
            "signatures": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": ["string", "null"]},
                        "conf": {"type": "string", "enum": ["unknown", "low", "medium", "high"]},
                        "source": {"type": "string", "enum": ["user", "tool", "llm", "inferred"]},
                        "evidence_snippet": {"type": ["string", "null"]},
                    },
                    "required": ["name", "value", "conf", "source", "evidence_snippet"],
                },
            },
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": ["string", "null"]},
                        "conf": {"type": "string", "enum": ["unknown", "low", "medium", "high"]},
                        "source": {"type": "string", "enum": ["user", "tool", "llm", "inferred"]},
                        "evidence_snippet": {"type": ["string", "null"]},
                    },
                    "required": ["name", "value", "conf", "source", "evidence_snippet"],
                },
            },
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["slots", "signatures", "entities", "notes"],
    }


EXTRACTION_SYSTEM = """You are an IT support triage extractor.

Goal: Extract structured fields from the user message for troubleshooting and KB retrieval.

Rules:
- Output MUST be valid JSON matching the provided schema.
- Only extract what is supported by the user message. Do NOT invent values.
- Use these slot names when possible:
  slots: product, application, plugin, os, version, network_location, managed_device, admin_rights, account_type
  signatures: error_signature, error_code, error_message
- conf guidance:
  - high: explicitly stated
  - medium: strongly implied
  - low: weak guess (avoid if possible)
  - unknown: leave value null
- evidence_snippet: quote a short phrase from the user text that supports the extraction.
"""

class _ExtractionResp(BaseModel):
    slots: List[ExtractedField] = Field(default_factory=list)
    signatures: List[ExtractedField] = Field(default_factory=list)
    entities: List[ExtractedField] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


def llm_extract_extractions(
    client: OpenAI,
    *,
    model: str,
    user_signal: UserSignal,
) -> ExtractionBundle:
    schema = extraction_json_schema()
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": f"USER MESSAGE:\n{user_signal.text}"},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "extractions",
                "strict": True,
                "schema": schema,
            }
        },
        temperature=0.1,
        max_output_tokens=900,
    )

    raw = getattr(resp, "output_text", "") or ""
    data = json.loads(raw)
    parsed = _ExtractionResp.model_validate(data)

    return ExtractionBundle(
        slots=parsed.slots,
        signatures=parsed.signatures,
        entities=parsed.entities,
        notes=parsed.notes,
    )


# =============================================================================
# 2) Retriever interface (you plug your own)
# =============================================================================

class Retriever:
    """
    Pluggable retriever interface.

    Implement retrieve() to return (ProcedureNode, retrieval_score) pairs.

    Your ProcedureNode type is from procedure_compiler.py.
    """
    def retrieve(self, *, query_text: str, frame: Dict[str, Any], top_k: int = 12) -> List[Tuple[Any, float]]:
        raise NotImplementedError


class InMemoryDemoRetriever(Retriever):
    """
    Simple demo retriever for quick bring-up.
    Replace with your hybrid retrieval (BM25 + embeddings + rerank).
    """
    def __init__(self, nodes: Sequence[Any]):
        self.nodes = list(nodes)

    def retrieve(self, *, query_text: str, frame: Dict[str, Any], top_k: int = 12) -> List[Tuple[Any, float]]:
        q = query_text.lower()
        out: List[Tuple[Any, float]] = []
        for n in self.nodes:
            text = ((getattr(n, "title", "") or "") + " " + " ".join(getattr(n, "signatures", []) or [])).lower()
            score = 0.0
            if q and any(tok in text for tok in q.split() if len(tok) > 2):
                score += 1.0
            # small boost if product mentioned
            prod = frame.get("product")
            if prod and getattr(prod, "value", None):
                if str(prod.value).lower() in text:
                    score += 0.5
            out.append((n, score))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_k]


# =============================================================================
# 3) Deterministic policy (ASK/ACT/VERIFY)
# =============================================================================

def choose_next_action(state: State) -> Action:
    """
    Deterministic action selection based on uncertainty + coverage.

    NOTE: You can expand this later (risk gating, verification failures, etc.)
    """
    b = state.belief.procedures
    u = state.uncertainty

    if not b:
        return Action(type="REQUEST_ARTIFACT", payload={"message": "Can you share the exact error message or a screenshot?"})

    top = b[0]

    # Contradictions: ask for the conflicting gatekeeper slot (V1 heuristic)
    if u.conflicts_top1 > 0:
        # Ask OS/app/product depending on what's missing
        # We determine missing from top hypothesis
        if "os" in top.missing_slots:
            return Action(type="ASK", payload={"target_slot": "os", "question": "Which OS are you on (Windows/macOS/Linux)?"})
        if "application" in top.missing_slots:
            return Action(type="ASK", payload={"target_slot": "application", "question": "Which app is affected (Excel, PowerPoint, Outlook, etc.)?"})
        if "product" in top.missing_slots or "plugin" in top.missing_slots:
            return Action(type="ASK", payload={"target_slot": "product", "question": "Which product/plugin is this issue about?"})

    # Uncertain: ask a discriminative question
    if u.top1 < 0.65 or u.margin < 0.15 or u.coverage_top1 < 0.7:
        # Ask for the highest-value missing slot for the top hypothesis
        if top.missing_slots:
            slot = top.missing_slots[0]
            qmap = {
                "error_signature": "What is the exact error message (or what do you see on screen)?",
                "error_message": "What is the exact error message (or what do you see on screen)?",
                "error_code": "What error code do you see (if any)?",
                "os": "Which OS are you on (Windows/macOS/Linux)?",
                "application": "Which app is affected (Excel/PowerPoint/Outlook/etc.)?",
                "product": "Which product/tool is this issue about?",
                "plugin": "Which plugin/add-in is involved?",
                "version": "What version are you on (app version if available)?",
                "network_location": "Are you on home Wi-Fi, office network, or VPN?",
            }
            return Action(type="ASK", payload={"target_slot": slot, "question": qmap.get(slot, f"Can you share {slot}?")})

        return Action(type="ASK", payload={"question": "Can you share the exact error message (or a screenshot)?"})

    # Confident: act
    return Action(type="ACT", payload={"procedure_id": top.procedure_id})


# =============================================================================
# 4) LLM response composition for ASK/ACT/VERIFY
# =============================================================================

ASK_SYSTEM = """You are an IT support agent.
Ask exactly ONE question. Keep it short and easy to answer.
Do not include multiple questions. Do not add extra commentary.
"""

ACT_SYSTEM = """You are an IT support agent.
Given a selected troubleshooting procedure, explain the steps clearly.
Rules:
- Only use steps provided (do not invent tools).
- Keep it concise but actionable.
- Include a short verification check at the end (one question).
"""

VERIFY_SYSTEM = """You are an IT support agent.
Ask a crisp verification question to confirm whether the previous steps worked.
If the user says 'no', ask for one key evidence item (error message or screenshot).
Keep it short.
"""


def llm_compose_ask(client: OpenAI, *, model: str, question: str) -> str:
    resp = client.responses.create(
        model=model,
        input=[{"role": "system", "content": ASK_SYSTEM}, {"role": "user", "content": question}],
        temperature=0.2,
        max_output_tokens=120,
    )
    return (getattr(resp, "output_text", "") or "").strip()


def llm_compose_act(
    client: OpenAI,
    *,
    model: str,
    procedure_node: Any,
    frame: Dict[str, Any],
) -> str:
    # Provide procedure content in a bounded way
    steps = getattr(procedure_node, "steps", []) or []
    step_lines: List[str] = []
    for i, st in enumerate(steps, start=1):
        instr = getattr(st, "instruction", "") or ""
        obs = getattr(st, "expected_observation", None)
        line = f"{i}. {instr}"
        if obs:
            line += f" (Expected: {obs})"
        step_lines.append(line)

    verify_prompts = getattr(procedure_node, "verify_prompts", []) or []
    verify = verify_prompts[0] if verify_prompts else "Did that resolve the issue?"

    user_content = f"""SELECTED PROCEDURE:
Title: {getattr(procedure_node, 'title', '')}
Summary: {getattr(procedure_node, 'summary', '')}

STEPS:
{chr(10).join(step_lines)}

VERIFY PROMPT:
{verify}

CONTEXT (slots):
{json.dumps({k: getattr(v, 'value', None) for k, v in frame.items()}, indent=2)}
"""

    resp = client.responses.create(
        model=model,
        input=[{"role": "system", "content": ACT_SYSTEM}, {"role": "user", "content": user_content}],
        temperature=0.3,
        max_output_tokens=400,
    )
    return (getattr(resp, "output_text", "") or "").strip()


def llm_compose_verify(client: OpenAI, *, model: str, verify_prompt: str) -> str:
    resp = client.responses.create(
        model=model,
        input=[{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": verify_prompt}],
        temperature=0.2,
        max_output_tokens=120,
    )
    return (getattr(resp, "output_text", "") or "").strip()


# =============================================================================
# 5) Orchestrator
# =============================================================================

class OrchestratorConfig(BaseModel):
    extractor_model: str = "gpt-4.1-mini"
    writer_model: str = "gpt-4.1-mini"
    top_k: int = 12
    belief_temperature: float = 1.0


class Orchestrator:
    def __init__(self, *, retriever: Retriever, config: Optional[OrchestratorConfig] = None, client: Optional[OpenAI] = None):
        self.retriever = retriever
        self.cfg = config or OrchestratorConfig()
        self.client = client or OpenAI()

        # You likely have a procedure store. For V1 we keep a local lookup cache.
        self._proc_by_id: Dict[str, Any] = {}

    def _index_candidates(self, candidates: Sequence[Tuple[Any, float]]) -> None:
        for node, _ in candidates:
            pid = str(getattr(node, "procedure_id", ""))
            if pid:
                self._proc_by_id[pid] = node

    def _compute_belief(self, *, frame_dict: Dict[str, Any], candidates: Sequence[Tuple[Any, float]]):
        from belief import build_belief  # your updated belief.py

        belief_list, unc = build_belief(
            frame=frame_dict,
            candidates=candidates,
            temperature=self.cfg.belief_temperature,
        )
        return belief_list, unc

    def handle_turn(self, state: Optional[State], user_signal: UserSignal) -> Tuple[str, State]:
        """
        Main entry:
          assistant_text, new_state = orchestrator.handle_turn(prev_state, user_signal)
        """
        if state is None:
            state = State(
                conversation_id=f"conv-{uuid.uuid4().hex[:8]}",
                turn_id=0,
                evidence=Evidence(user_signal=user_signal),
            )

        # Increment turn & set evidence.signal
        state.turn_id += 1
        state.evidence.user_signal = user_signal

        # 1) LLM extraction
        extractions = llm_extract_extractions(self.client, model=self.cfg.extractor_model, user_signal=user_signal)
        state.evidence.extractions = extractions

        # 2) Convert extractions -> frame mapping for belief
        frame = extraction_to_frame(extractions)  # Dict[str, SlotValue] compatible with belief.py
        # belief.py expects its own SlotValue dataclass; but it only reads .value/.conf
        # Our SlotValue in state_schema matches that interface, so it works.

        # 3) Retrieve candidates
        candidates = self.retriever.retrieve(query_text=user_signal.text, frame=frame, top_k=self.cfg.top_k)
        self._index_candidates(candidates)

        # Also store retrieval candidates in evidence
        rc: List[RetrievalCandidate] = []
        for node, rscore in candidates:
            rc.append(
                RetrievalCandidate(
                    procedure_id=str(getattr(node, "procedure_id", "")),
                    retrieval_score=float(rscore),
                    doc_id=str(getattr(node, "doc_id", "")) if getattr(node, "doc_id", None) else None,
                    doc_title=str(getattr(node, "doc_title", "")) if getattr(node, "doc_title", None) else None,
                    title=str(getattr(node, "title", "")) if getattr(node, "title", None) else None,
                )
            )
        state.evidence.retrieval_candidates = rc

        # 4) Belief + uncertainty deterministic
        belief_list, unc = self._compute_belief(frame_dict=frame, candidates=candidates)

        # Map belief.py dataclasses -> state_schema models
        belief_models: List[BeliefObjCandidate] = []
        for b in belief_list:
            belief_models.append(
                BeliefObjCandidate(
                    procedure_id=b.procedure_id,
                    prob=b.prob,
                    score=b.score,
                    score_breakdown=ScoreBreakdownObj(**asdict(b.score_breakdown)),
                    missing_slots=b.missing_slots,
                    conflicts=b.conflicts,
                    risk=b.risk,  # type: ignore
                    signature_strength=b.signature_strength,  # type: ignore
                )
            )
        state.belief = BeliefObj(procedures=belief_models)

        state.uncertainty = Uncertainty(
            entropy=unc.entropy,
            entropy_norm=unc.entropy_norm,
            top1=unc.top1,
            top2=unc.top2,
            margin=unc.margin,
            coverage_top1=unc.coverage_top1,
            conflicts_top1=unc.conflicts_top1,
            signature_strength_top1=unc.signature_strength_top1,  # type: ignore
        )

        # 5) Choose action deterministically
        action = choose_next_action(state)
        state.last_action = action

        # 6) Compose assistant message via LLM (or deterministic fallback)
        assistant_text = self._compose(action=action, state=state, frame=frame)

        # 7) Append compact history event
        state.history.append(
            {
                "turn_id": state.turn_id,
                "user_text": user_signal.text,
                "action": action.model_dump(),
                "uncertainty": state.uncertainty.model_dump(),
                "top_procedure": state.belief.procedures[0].procedure_id if state.belief.procedures else None,
            }
        )

        return assistant_text, state

    def _compose(self, *, action: Action, state: State, frame: Dict[str, Any]) -> str:
        if action.type == "ASK":
            q = action.payload.get("question", "Can you share more details?")
            return llm_compose_ask(self.client, model=self.cfg.writer_model, question=q)

        if action.type == "ACT":
            pid = action.payload.get("procedure_id")
            node = self._proc_by_id.get(pid)
            if not node:
                # fallback: ask for more info
                return llm_compose_ask(
                    self.client,
                    model=self.cfg.writer_model,
                    question="I couldn’t find the exact runbook step. What is the exact error message or screenshot?"
                )
            # Compose steps
            act_text = llm_compose_act(self.client, model=self.cfg.writer_model, procedure_node=node, frame=frame)

            # In V1, we auto-transition to VERIFY next turn based on user's reply.
            # (You can store a pending_verify flag here if you want.)
            return act_text

        if action.type == "REQUEST_ARTIFACT":
            msg = action.payload.get("message", "Can you share the exact error message or a screenshot?")
            return llm_compose_ask(self.client, model=self.cfg.writer_model, question=msg)

        if action.type == "ESCALATE":
            return "This likely needs escalation. Please share the error message and a screenshot, and I’ll package the details for the support team."

        return "Okay—tell me what you’re seeing and I’ll help."


# =============================================================================
# 6) Minimal CLI chat loop
# =============================================================================

def cli_chat_loop(orchestrator: Orchestrator) -> None:
    state: Optional[State] = None
    print("IT Support Agent (V1) — type 'exit' to quit.\n")

    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break

        user_signal = UserSignal(type="chat", text=user)
        assistant_text, state = orchestrator.handle_turn(state, user_signal)
        print(f"\nAssistant: {assistant_text}\n")

        # Optional: show debug summary
        if state.belief.procedures:
            top = state.belief.procedures[0]
            u = state.uncertainty
            print(f"[debug] top={top.procedure_id} p={top.prob:.2f} margin={u.margin:.2f} cov={u.coverage_top1:.2f} Hn={u.entropy_norm:.2f}\n")


# =============================================================================
# 7) Demo wiring (replace with your real retriever / node store)
# =============================================================================

if __name__ == "__main__":
    # Demo only: create 2 fake nodes shaped like your ProcedureNode
    class _PC:
        def __init__(self, slot: str, values: List[str]):
            self.slot = slot
            self.values = values

    class _Step:
        def __init__(self, instruction: str, expected_observation: Optional[str] = None):
            self.instruction = instruction
            self.expected_observation = expected_observation

    class _Node:
        def __init__(self, procedure_id: str, title: str, signatures: List[str], preconditions, required_slots, steps, verify_prompts, tags, risk="low"):
            self.procedure_id = procedure_id
            self.doc_id = "kb-123"
            self.doc_title = "Procedure - ThinkCell"
            self.title = title
            self.summary = "Basic troubleshooting steps."
            self.signatures = signatures
            self.preconditions = preconditions
            self.required_slots = required_slots
            self.steps = steps
            self.verify_prompts = verify_prompts
            self.tags = tags
            self.risk = risk

    n1 = _Node(
        "proc:thinkcell_basic",
        "Basic troubleshooting for ThinkCell plugin issues",
        signatures=["plugin not working", "add-in missing", "ThinkCell issue"],
        preconditions=[_PC("application", ["Excel", "PowerPoint"])],
        required_slots=["application", "product"],
        steps=[
            _Step("Open Excel or PowerPoint."),
            _Step("Go to File > Options > Add-ins."),
            _Step("Manage: COM Add-ins > Go, uncheck ThinkCell, click OK."),
            _Step("Restart Excel/PowerPoint."),
            _Step("Repeat to re-enable ThinkCell."),
        ],
        verify_prompts=["Verify the ThinkCell tab/ribbon appears."],
        tags=["thinkcell", "plugin", "powerpoint", "excel"],
        risk="low",
    )

    n2 = _Node(
        "proc:thinkcell_escalate",
        "Escalate ThinkCell issues after basic troubleshooting",
        signatures=["escalate to support team", "unresolved after basic troubleshooting"],
        preconditions=[_PC("product", ["ThinkCell"])],
        required_slots=["product", "application"],
        steps=[
            _Step("Collect error screenshot and error message."),
            _Step("Collect user contact details and preferred contact method."),
            _Step("Escalate to End User Platform – Windows with gathered evidence."),
        ],
        verify_prompts=["Confirm escalation details are complete."],
        tags=["thinkcell", "escalation"],
        risk="low",
    )

    retriever = InMemoryDemoRetriever([n1, n2])
    orch = Orchestrator(retriever=retriever, config=OrchestratorConfig())

    cli_chat_loop(orch)
