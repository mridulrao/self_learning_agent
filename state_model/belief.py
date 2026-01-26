#!/usr/bin/env python3
"""
belief.py

Deterministic belief computation over your ProcedureNode objects from procedure_compiler.py
(where:
  - preconditions: List[Precondition(slot, values)]
  - required_slots: List[str]
  - signatures: List[str] (may include soft signatures)
  - steps: List[ProcedureStep(instruction, ...))
  - tags: List[str]
  - risk: "low"|"medium"|"high"
)

Given:
  - frame: extracted slots (value + confidence)
  - candidates: [(procedure_node, retrieval_score), ...]  # from your hybrid retriever

Compute:
  - belief distribution b(h) over procedure hypotheses (softmax over combined scores)
  - uncertainty metrics (entropy, margin, coverage, conflicts, signature strength)
  - per-candidate explainable score breakdown

No external deps (pure stdlib).

Usage:
  from belief import SlotValue, build_belief
  belief, unc = build_belief(frame=frame, candidates=candidates)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple


# -----------------------------
# Frame schema (flexible)
# -----------------------------

SlotConf = Literal["unknown", "low", "medium", "high"]

@dataclass
class SlotValue:
    value: Optional[str] = None
    conf: SlotConf = "unknown"
    source: Optional[str] = None  # "user" | "tool" | "llm" | etc.

Frame = Dict[str, SlotValue]


# -----------------------------
# Belief outputs
# -----------------------------

SigStrength = Literal["none", "weak", "regex", "exact"]

@dataclass
class ScoreBreakdown:
    retrieval: float = 0.0
    signature_match: float = 0.0
    applicability: float = 0.0
    slot_alignment: float = 0.0
    total: float = 0.0

@dataclass
class BeliefCandidate:
    procedure_id: str
    prob: float
    score: float
    score_breakdown: ScoreBreakdown
    missing_slots: List[str]
    conflicts: List[str]
    risk: str = "low"
    signature_strength: SigStrength = "none"

@dataclass
class UncertaintyMetrics:
    top1: float = 0.0
    top2: float = 0.0
    margin: float = 0.0
    entropy: float = 0.0
    entropy_norm: float = 0.0
    coverage_top1: float = 0.0
    conflicts_top1: int = 0
    signature_strength_top1: SigStrength = "none"


# -----------------------------
# Helpers: normalization, softmax, entropy
# -----------------------------

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _tokenize(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_]+", s.lower()) if t]

def slot_filled(frame: Frame, slot: str, min_conf: Tuple[SlotConf, ...] = ("medium", "high")) -> bool:
    sv = frame.get(slot)
    return bool(sv and sv.value is not None and sv.conf in min_conf and str(sv.value).strip() != "")

def compute_coverage(frame: Frame, required_slots: Sequence[str]) -> float:
    req = [s for s in required_slots if isinstance(s, str) and s.strip()]
    if not req:
        return 1.0
    filled = sum(1 for s in req if slot_filled(frame, s))
    return filled / max(len(req), 1)

def softmax(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    t = max(float(temperature), 1e-6)
    m = max(scores)
    exps = [math.exp((s - m) / t) for s in scores]
    z = sum(exps) or 1.0
    return [e / z for e in exps]

def entropy(probs: Sequence[float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(p + eps) for p in probs)

def norm_entropy(probs: Sequence[float]) -> float:
    if len(probs) <= 1:
        return 0.0
    return entropy(probs) / math.log(len(probs))

def _get_node_text(node: Any) -> str:
    """
    Aggregate text from your ProcedureNode:
      - title, summary
      - tags
      - step instructions
      - signatures
    """
    parts: List[str] = []
    for k in ("title", "summary"):
        v = getattr(node, k, None)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    tags = getattr(node, "tags", None)
    if isinstance(tags, list):
        parts.extend([str(t).strip() for t in tags if str(t).strip()])

    sigs = getattr(node, "signatures", None)
    if isinstance(sigs, list):
        parts.extend([str(s).strip() for s in sigs if str(s).strip()])

    steps = getattr(node, "steps", None)
    if isinstance(steps, list):
        for st in steps:
            instr = getattr(st, "instruction", None)
            if isinstance(instr, str) and instr.strip():
                parts.append(instr.strip())
            obs = getattr(st, "expected_observation", None)
            if isinstance(obs, str) and obs.strip():
                parts.append(obs.strip())

    return "\n".join(parts).lower()


# -----------------------------
# Signature matching
# -----------------------------

def signature_score(
    frame: Frame,
    node: Any,
    *,
    frame_sig_slots: Sequence[str] = ("error_signature", "error_code", "error_message"),
) -> Tuple[SigStrength, float]:
    """
    Returns (strength, score).

    Works with your node.signatures which can contain:
      - exact error strings/codes
      - regex patterns
      - soft signatures (short phrases)

    Heuristic:
      - exact substring match of frame signature in node text -> exact (high)
      - regex pattern from node.signatures matches -> regex (medium)
      - token overlap with node.signatures/text -> weak (low)
    """
    sig_val: Optional[str] = None
    for s in frame_sig_slots:
        sv = frame.get(s)
        if sv and sv.value and str(sv.value).strip():
            sig_val = str(sv.value).strip()
            break

    if not sig_val:
        return "none", 0.0

    sig_raw = sig_val
    sig = sig_raw.lower().strip()
    text = _get_node_text(node)

    # Exact match (strong)
    if len(sig) >= 4 and sig in text:
        return "exact", 4.0

    # Regex match (medium)
    patterns = getattr(node, "signatures", None)
    if isinstance(patterns, list):
        for pat in patterns:
            if not isinstance(pat, str) or not pat.strip():
                continue
            # Treat only "regex-looking" strings as patterns to avoid accidental regex explosions
            # If user stores plain strings, exact is already handled above.
            looks_regex = bool(re.search(r"[\\\[\]\(\)\|\.\+\*\?\^\$]", pat))
            if not looks_regex:
                continue
            try:
                if re.search(pat, sig_raw, re.IGNORECASE):
                    return "regex", 2.5
            except re.error:
                continue

    # Weak match (token overlap) - especially useful when sig_raw is not provided as exact
    sig_toks = [t for t in _tokenize(sig) if len(t) > 3]
    if not sig_toks:
        return "none", 0.0

    overlap = sum(1 for t in sig_toks if t in text)
    if overlap >= 3:
        return "weak", 1.2
    if overlap == 2:
        return "weak", 1.0
    if overlap == 1:
        return "weak", 0.5

    return "none", 0.0


# -----------------------------
# Applicability / Preconditions scoring (UPDATED for your Precondition list)
# -----------------------------

def applicability_score(frame: Frame, node: Any) -> Tuple[float, List[str]]:
    """
    Score how well node.preconditions match frame.

    Your node.preconditions: List[Precondition(slot, values)]
    where values is List[str].

    Heuristic:
      - If frame has value and it is in allowed values -> +1.0
      - If frame has value and NOT in allowed -> conflict + -0.8
      - If unknown -> 0
    """
    pre = getattr(node, "preconditions", None)
    if not isinstance(pre, list) or not pre:
        return 0.0, []

    score = 0.0
    conflicts: List[str] = []

    for pc in pre:
        slot = _norm(getattr(pc, "slot", None))
        if not slot:
            continue

        allowed = getattr(pc, "values", None)
        if not isinstance(allowed, list):
            continue

        allowed_norm = [_norm(str(x)) for x in allowed if str(x).strip()]
        if not allowed_norm:
            continue

        sv = frame.get(slot)
        if sv and sv.value and str(sv.value).strip():
            v = _norm(str(sv.value))

            # Allow a small normalization for common OS variants
            v2 = _normalize_common_values(slot, v)
            allowed2 = [_normalize_common_values(slot, a) for a in allowed_norm]

            if v2 in allowed2:
                score += 1.0
            else:
                conflicts.append(f"{slot} mismatch: {sv.value} not in {allowed}")
                score -= 0.8
        else:
            score += 0.0

    return score, conflicts


def _normalize_common_values(slot: str, val: str) -> str:
    """
    Small normalizer for common slots (esp. os).
    Keep conservative; don't invent new mappings.
    """
    if slot == "os":
        if val in ("mac", "macos", "osx", "darwin"):
            return "macos"
        if val in ("win", "windows", "win10", "win11"):
            return "windows"
        if val in ("linux", "ubuntu", "debian", "fedora", "centos", "rhel"):
            return "linux"
    return val


# -----------------------------
# Slot/entity alignment scoring
# -----------------------------

def slot_alignment_score(
    frame: Frame,
    node: Any,
    *,
    slots: Sequence[str] = ("product", "application", "plugin", "os"),
) -> float:
    """
    Small heuristic bonus if key slot values are mentioned in the node text.
    This helps when retrieval is close and signatures are soft.

    IMPORTANT: keep this weak; signature/applicability should dominate.
    """
    text = _get_node_text(node)
    s = 0.0
    for slot in slots:
        sv = frame.get(slot)
        if sv and sv.value and str(sv.value).strip():
            token = _norm(str(sv.value))
            if token and token in text:
                if slot in ("product", "application", "plugin"):
                    s += 0.4
                elif slot == "os":
                    s += 0.25
                else:
                    s += 0.2
    return s


# -----------------------------
# Belief builder (main)
# -----------------------------

def build_belief(
    *,
    frame: Frame,
    candidates: Sequence[Tuple[Any, float]],  # (ProcedureNode, retrieval_score)
    weights: Optional[Dict[str, float]] = None,
    temperature: float = 1.0,
    top_n: int = 20,
) -> Tuple[List[BeliefCandidate], UncertaintyMetrics]:
    """
    Compute belief distribution over candidate procedure nodes.
    """
    w = weights or {
        "retrieval": 1.0,
        "signature": 1.6,
        "applicability": 1.0,
        "slot": 0.6,
    }

    scored_totals: List[float] = []
    metas: List[Tuple[Any, ScoreBreakdown, List[str], List[str], SigStrength]] = []

    for node, rscore in candidates[: max(top_n, 1)]:
        sig_strength, sig = signature_score(frame, node)
        app, conflicts = applicability_score(frame, node)
        slot = slot_alignment_score(frame, node)

        total = (
            w["retrieval"] * float(rscore)
            + w["signature"] * float(sig)
            + w["applicability"] * float(app)
            + w["slot"] * float(slot)
        )

        sb = ScoreBreakdown(
            retrieval=w["retrieval"] * float(rscore),
            signature_match=w["signature"] * float(sig),
            applicability=w["applicability"] * float(app),
            slot_alignment=w["slot"] * float(slot),
            total=total,
        )

        required = getattr(node, "required_slots", []) or []
        missing = [s for s in required if isinstance(s, str) and s.strip() and not slot_filled(frame, s)]

        metas.append((node, sb, missing, conflicts, sig_strength))
        scored_totals.append(total)

    probs = softmax(scored_totals, temperature=temperature)

    belief: List[BeliefCandidate] = []
    for (node, sb, missing, conflicts, sig_strength), p, s in zip(metas, probs, scored_totals):
        belief.append(
            BeliefCandidate(
                procedure_id=str(getattr(node, "procedure_id", "")),
                prob=float(p),
                score=float(s),
                score_breakdown=sb,
                missing_slots=missing,
                conflicts=conflicts,
                risk=str(getattr(node, "risk", "low")),
                signature_strength=sig_strength,
            )
        )

    belief.sort(key=lambda x: x.prob, reverse=True)

    # Uncertainty metrics
    ps = [b.prob for b in belief]
    top1 = ps[0] if ps else 0.0
    top2 = ps[1] if len(ps) > 1 else 0.0
    marg = top1 - top2
    H = entropy(ps) if ps else 0.0
    Hn = norm_entropy(ps) if ps else 0.0

    coverage_top1 = 0.0
    conflicts_top1 = 0
    sig_strength_top1: SigStrength = "none"

    if belief:
        top_id = belief[0].procedure_id
        top_node = None
        top_sig_strength = "none"
        for (node, _sb, _missing, _conflicts, _ss) in metas:
            if str(getattr(node, "procedure_id", "")) == top_id:
                top_node = node
                top_sig_strength = _ss
                break

        if top_node is not None:
            req = getattr(top_node, "required_slots", []) or []
            coverage_top1 = compute_coverage(frame, [s for s in req if isinstance(s, str)])

        conflicts_top1 = len(belief[0].conflicts)
        sig_strength_top1 = top_sig_strength  # type: ignore

    unc = UncertaintyMetrics(
        top1=float(top1),
        top2=float(top2),
        margin=float(marg),
        entropy=float(H),
        entropy_norm=float(Hn),
        coverage_top1=float(coverage_top1),
        conflicts_top1=int(conflicts_top1),
        signature_strength_top1=sig_strength_top1,
    )

    return belief, unc


# -----------------------------
# Demo (optional)
# -----------------------------

if __name__ == "__main__":
    # This demo expects your ProcedureNode type, but we’ll create small shims.
    class _PC:
        def __init__(self, slot: str, values: List[str]):
            self.slot = slot
            self.values = values

    class _Step:
        def __init__(self, instruction: str, expected_observation: Optional[str] = None):
            self.instruction = instruction
            self.expected_observation = expected_observation

    class _Node:
        def __init__(
            self,
            procedure_id: str,
            title: str,
            signatures: List[str],
            preconditions: List[_PC],
            required_slots: List[str],
            steps: List[str],
            tags: List[str] = None,
            risk: str = "low",
        ):
            self.procedure_id = procedure_id
            self.title = title
            self.summary = ""
            self.signatures = signatures
            self.preconditions = preconditions
            self.required_slots = required_slots
            self.steps = [_Step(s) for s in steps]
            self.tags = tags or []
            self.risk = risk

    frame: Frame = {
        "product": SlotValue("ThinkCell", "high", "user"),
        "application": SlotValue("PowerPoint", "high", "user"),
        "os": SlotValue("windows", "medium", "user"),
        "error_signature": SlotValue("ThinkCell tab missing", "medium", "user"),
    }

    n1 = _Node(
        "proc:thinkcell_basic_troubleshoot",
        "Basic troubleshooting for ThinkCell plugin issues",
        signatures=["plugin not working", "add-in missing", "ThinkCell issue"],
        preconditions=[_PC("application", ["Excel", "PowerPoint"])],
        required_slots=["application", "product"],
        steps=["Open PowerPoint", "Disable ThinkCell add-in", "Restart PowerPoint", "Re-enable ThinkCell add-in"],
        tags=["plugin", "thinkcell", "powerpoint"],
        risk="low",
    )
    n2 = _Node(
        "proc:thinkcell_install",
        "Install ThinkCell from corporate package",
        signatures=["installation", "setup", "installer"],
        preconditions=[_PC("os", ["windows"])],
        required_slots=["os", "product"],
        steps=["Download installer", "Run installer as admin", "Verify ThinkCell tab appears"],
        tags=["install", "thinkcell"],
        risk="medium",
    )

    # pretend hybrid retrieval scores
    candidates = [(n1, 2.0), (n2, 1.8)]

    belief, unc = build_belief(frame=frame, candidates=candidates)
    print("UNCERTAINTY:", asdict(unc))
    print("TOP BELIEF:", [asdict(b) for b in belief[:2]])
