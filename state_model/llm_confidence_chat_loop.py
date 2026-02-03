#!/usr/bin/env python3
"""
llm_confidence_chat_loop.py

A production-lean architecture for "uncertainty -> confidence -> doc lane execution"
controller using TWO LLM calls per turn:

1) Extractor LLM:
   Raw user input (chat/voice transcript + artifacts) -> ExtractionTrace + ExtractedFacts

2) Controller LLM (proposal layer):
   Uses ControllerState snapshot + candidate lanes + extracted facts -> proposes:
   - belief deltas (lane score nudges)
   - next action (ASK_EVIDENCE / EXEC_STEP / SWITCH_LANE / ESCALATE / RESOLVE)
   - debug rationale

Authoritative code responsibilities (IMPORTANT):
- merges facts deterministically
- applies deterministic belief updates from:
  - retrieval priors (ctx["retrieval_priors"])
  - HYBRID FACT MATCHING (BM25 + semantic + optional cross-encoder)
  - fact/lane compatibility (fact_specs + preconditions + signatures)
  - step validations (when executing)
- normalizes belief distribution (softmax)
- enforces budgets + hysteresis switching
- evaluates step pass/fail + transitions deterministically
- validates IDs and step transitions
- produces TurnTrace for debuggability

RECENT CHANGES (2024 - Hybrid Retrieval):
- Integrated BM25 + semantic embeddings + cross-encoder for fact-lane matching
- Replaced hardcoded fact_specs matching with flexible hybrid retrieval
- Handles synonyms, paraphrasing, and technical terms automatically
- Adaptive phase-based capping (looser in TRIAGE, tighter in EXECUTE)
- Cross-encoder reranking for high-accuracy disambiguation when needed
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from pydantic_models import (
    ActionType,
    AgentTurnResult,
    BeliefDelta,
    BeliefDeltaProposal,
    BeliefItem,
    BeliefState,
    BeliefTrace,
    Comparator,
    Condition,
    ConstraintSet,
    ControllerProposal,
    ControllerState,
    EvidenceAction,
    EvidenceItem,
    EvidenceRequest,
    ExtractedFact,
    ExtractionTrace,
    ExtractorOutput,
    FactSpec,
    Phase,
    PolicyTrace,
    ProcedureLane,
    ProcedureStep,
    StepEvalResult,
    StepTrace,
    TurnTrace,
    _canon_key,
)

# Import hybrid retrieval
from hybrid_fact_matching import (
    compute_embeddings_batch,
    compute_hybrid_match_score,
    cross_encoder_rerank,
    format_facts_as_query,
    format_match_reason,
    match_score_to_belief_delta,
)


# =============================================================================
# Utility functions
# =============================================================================

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _softmax(
    scores: Dict[str, float], 
    temperature: float = 0.7,
    turn_count: int = 0,
    phase: str = "triage",
    current_top_prob: float = 0.0,
) -> Dict[str, float]:
    """Softmax with adaptive temperature."""
    if not scores:
        return {}
    
    # Use adaptive temp if turn_count > 0
    if turn_count > 0:
        temperature = _get_adaptive_temperature(turn_count, phase, current_top_prob)
    
    t = max(float(temperature), 1e-6)
    m = max(scores.values())
    exps = {k: pow(2.718281828, (v - m) / t) for k, v in scores.items()}
    z = sum(exps.values()) or 1.0
    return {k: (v / z) for k, v in exps.items()}


def _constraints_empty(cs: ConstraintSet) -> bool:
    return (not cs.all_of) and (not cs.any_of) and (not cs.none_of)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _get_adaptive_temperature(
    turn_count: int,
    phase: str,
    current_top_prob: float = 0.0,
) -> float:
    """
    Adaptive temperature that sharpens distribution over time.
    
    Key principle:
    - High temp (0.7) initially = broad exploration
    - Decrease temp each turn = sharpen distribution
    - React to confidence = sharpen faster when leader emerges
    
    Args:
        turn_count: Number of turns so far
        phase: "triage" or "execute"
        current_top_prob: Highest probability in current distribution
        
    Returns:
        Temperature value for softmax
    """
    if phase.lower() == "triage":
        # Progressive annealing: 0.7 → 0.35 over 4 turns
        base_temp = max(0.35, 0.7 - (turn_count * 0.09))
        
        # If we're close to threshold, sharpen aggressively
        if current_top_prob >= 0.45:
            base_temp *= 0.8
        
        return base_temp
    else:
        # Sharp distribution during execution
        return 0.25


# =============================================================================
# Condition evaluation
# =============================================================================

def _get_path_value(
    path: str,
    facts: Dict[str, ExtractedFact],
    slots: Dict[str, Any],
    artifacts: Dict[str, Any],
    ctx: Dict[str, Any],
    belief_signals: Dict[str, Any],
) -> Any:
    if path.startswith("fact."):
        k = _canon_key(path[len("fact.") :])
        return facts.get(k).value if k in facts else None
    if path.startswith("slot."):
        k = _canon_key(path[len("slot.") :])
        return slots.get(k)
    if path.startswith("artifact."):
        k = _canon_key(path[len("artifact.") :])
        return artifacts.get(k)
    if path.startswith("ctx."):
        k = path[len("ctx.") :]
        return ctx.get(k)
    if path.startswith("belief."):
        k = path[len("belief.") :]
        return belief_signals.get(k)
    return None


def _eval_condition(
    cond: Condition,
    facts: Dict[str, ExtractedFact],
    slots: Dict[str, Any],
    artifacts: Dict[str, Any],
    ctx: Dict[str, Any],
    belief_signals: Dict[str, Any],
) -> bool:
    lhs_val = _get_path_value(cond.lhs, facts, slots, artifacts, ctx, belief_signals)

    op = cond.op
    rhs = cond.rhs

    if op == Comparator.EXISTS:
        return lhs_val is not None
    if op == Comparator.NOT_EXISTS:
        return lhs_val is None

    if op == Comparator.EQ:
        return lhs_val == rhs
    if op == Comparator.NEQ:
        return lhs_val != rhs

    if op == Comparator.CONTAINS:
        if lhs_val is None:
            return False
        return str(rhs).lower() in str(lhs_val).lower()

    if op == Comparator.REGEX:
        if lhs_val is None or rhs is None:
            return False
        try:
            return re.search(str(rhs), str(lhs_val)) is not None
        except re.error:
            return False

    if op == Comparator.IN:
        if rhs is None:
            return False
        try:
            return lhs_val in rhs
        except TypeError:
            return False

    if op == Comparator.NIN:
        if rhs is None:
            return True
        try:
            return lhs_val not in rhs
        except TypeError:
            return True

    if op == Comparator.GTE:
        try:
            return float(lhs_val) >= float(rhs)
        except Exception:
            return False

    if op == Comparator.LTE:
        try:
            return float(lhs_val) <= float(rhs)
        except Exception:
            return False

    return False


def eval_constraints(
    cs: ConstraintSet,
    facts: Dict[str, ExtractedFact],
    slots: Dict[str, Any],
    artifacts: Dict[str, Any],
    ctx: Dict[str, Any],
    belief_signals: Dict[str, Any],
) -> bool:
    for c in cs.all_of:
        if not _eval_condition(c, facts, slots, artifacts, ctx, belief_signals):
            return False
    for c in cs.none_of:
        if _eval_condition(c, facts, slots, artifacts, ctx, belief_signals):
            return False
    if cs.any_of:
        ok = any(_eval_condition(c, facts, slots, artifacts, ctx, belief_signals) for c in cs.any_of)
        if not ok:
            return False
    return True


# =============================================================================
# Fact merging
# =============================================================================

def merge_facts(known: Dict[str, ExtractedFact], new_facts: List[ExtractedFact]) -> Dict[str, ExtractedFact]:
    """
    Deterministic merge:
      - overwrite if new confidence >= old confidence (or old missing)
      - otherwise keep old
    """
    out = dict(known)
    for nf in new_facts:
        nf.fact = _canon_key(nf.fact)
        old = out.get(nf.fact)
        if old is None or nf.confidence >= old.confidence:
            out[nf.fact] = nf
    return out


# =============================================================================
# Progress cost calculation
# =============================================================================

def compute_progress_switch_cost(steps_executed_in_lane: int, max_cost: float) -> float:
    cost = 0.05 + 0.03 * max(0, steps_executed_in_lane)
    return min(cost, max_cost)


# =============================================================================
# Deterministic belief updates (HYBRID RETRIEVAL INTEGRATION)
# =============================================================================

def _sig_match(signatures: List[str], text: str) -> bool:
    if not signatures or not text:
        return False
    t = str(text).lower()
    for s in signatures[:30]:
        ss = (s or "").strip().lower()
        if not ss:
            continue
        if ss in t:
            return True
    return False


def compute_lane_fact_delta_legacy(
    lane: ProcedureLane,
    facts: Dict[str, ExtractedFact],
    phase: str = "triage",
) -> Tuple[float, str]:
    """
    LEGACY exact-match fact scoring (kept as fallback).
    
    Only used if lane has no hybrid indices (old compiled lanes).
    """
    if not lane.fact_specs:
        return 0.0, "no_fact_specs"

    required = [fs for fs in lane.fact_specs if fs.required_for_certainty]
    optional = [fs for fs in lane.fact_specs if not fs.required_for_certainty]

    def present(fs: FactSpec) -> bool:
        k = _canon_key(fs.fact)
        return k in facts and facts[k].value is not None

    req_present = sum(1 for fs in required if present(fs))
    opt_present = sum(1 for fs in optional[:20] if present(fs))

    req_total = max(1, len(required))
    req_cov = req_present / req_total

    # confidence-weighted coverage
    conf_bonus = 0.0
    for fs in required[:25]:
        k = _canon_key(fs.fact)
        if k in facts and facts[k].value is not None:
            conf_bonus += float(facts[k].confidence) * (0.4 / req_total)

    # precondition compatibility
    precond_pen = 0.0
    for pc in lane.preconditions[:20]:
        slot = _canon_key(pc.slot)
        if not slot or slot not in facts:
            continue
        v = facts[slot].value
        if v is None:
            continue
        vals = [str(x).strip().lower() for x in (pc.values or []) if str(x).strip()]
        if vals:
            if str(v).strip().lower() not in vals:
                precond_pen += 0.15

    # signatures boost
    err_text = ""
    if "error_message" in facts and facts["error_message"].value:
        err_text += " " + str(facts["error_message"].value)
    if "error_code" in facts and facts["error_code"].value:
        err_text += " " + str(facts["error_code"].value)
    sig_boost = 0.08 if _sig_match(lane.signatures or [], err_text) else 0.0

    delta = 0.15 * req_cov + 0.05 * (opt_present / 10.0) + 0.10 * conf_bonus + sig_boost - precond_pen
    
    # Phase-based capping
    if phase.lower() == "triage":
        delta = max(-0.35, min(0.35, delta))
    else:
        delta = max(-0.20, min(0.20, delta))

    reason = f"legacy_exact req={req_present}/{req_total} opt={opt_present} precond_pen={precond_pen:.2f}"
    return delta, reason


# Global cache for query embeddings (to avoid redundant API calls)
_query_embedding_cache: Dict[str, List[float]] = {}


def compute_lane_fact_delta_hybrid(
    lane: ProcedureLane,
    facts: Dict[str, ExtractedFact],
    phase: str = "triage",
    api_key: Optional[str] = None,
    use_cross_encoder: bool = False,
) -> Tuple[float, str]:
    """
    HYBRID fact-to-lane matching using BM25 + semantic + optional cross-encoder.
    
    This is the NEW primary matching strategy.
    
    Args:
        lane: ProcedureLane with hybrid indices (fact_keywords, fact_embeddings)
        facts: Extracted facts from user
        phase: "triage" or "execute" (affects delta capping)
        api_key: OpenAI API key for embedding computation
        use_cross_encoder: Whether to use cross-encoder reranking
        
    Returns:
        (delta, reason) where delta is scaled based on hybrid match score
    """
    if not facts:
        return 0.0, "no_facts"
    
    # Check if lane has hybrid indices
    has_keywords = hasattr(lane, 'fact_keywords') and lane.fact_keywords
    has_embeddings = hasattr(lane, 'fact_embeddings') and lane.fact_embeddings
    
    if not has_keywords and not has_embeddings:
        # Fallback to legacy exact matching
        return compute_lane_fact_delta_legacy(lane, facts, phase)
    
    # Build lane retrieval index from lane attributes
    lane_index = {
        "fact_keywords": getattr(lane, 'fact_keywords', []),
        "fact_embeddings": getattr(lane, 'fact_embeddings', []),
        "fact_descriptions": getattr(lane, 'fact_descriptions', []),
    }
    
    # Compute query embedding (cached per query text)
    query_text, query_tokens = format_facts_as_query(facts)
    
    query_embedding = None
    if api_key and has_embeddings:
        # Check cache first
        cache_key = f"qemb:{query_text[:100]}"
        
        if cache_key in _query_embedding_cache:
            query_embedding = _query_embedding_cache[cache_key]
        else:
            # Compute fresh
            embeddings = compute_embeddings_batch(
                [query_text],
                embedding_model="text-embedding-3-small",
                api_key=api_key
            )
            if embeddings:
                query_embedding = embeddings[0]
                _query_embedding_cache[cache_key] = query_embedding
    
    # === Stage 1 & 2: BM25 + Semantic Fusion ===
    match_score, component_scores = compute_hybrid_match_score(
        facts=facts,
        lane_index=lane_index,
        query_embedding=query_embedding,
        fusion_method="weighted",
        weights={"bm25": 0.3, "semantic": 0.7},  # Favor semantic for concept matching
    )
    
    # === Stage 3: Optional Cross-Encoder Reranking ===
    reranked_score = None
    
    if use_cross_encoder and lane_index.get("fact_descriptions"):
        # Only rerank if we have enough context
        ce_results = cross_encoder_rerank(
            query=query_text,
            candidates=[(lane.lane_id, " ".join(lane_index["fact_descriptions"][:5]))],
            top_k=1,
        )
        
        if ce_results:
            reranked_score = ce_results.get(lane.lane_id)
            
            # Blend: 40% fusion + 60% cross-encoder (trust cross-encoder more)
            if reranked_score is not None:
                match_score = 0.4 * match_score + 0.6 * reranked_score
    
    # === Convert match score to belief delta ===
    delta = match_score_to_belief_delta(match_score, phase=phase)
    
    # Format reason for debugging
    reason = format_match_reason(component_scores, match_score, reranked_score)
    reason = f"hybrid[{reason}]"
    
    return delta, reason


def compute_lane_fact_delta(
    lane: ProcedureLane,
    facts: Dict[str, ExtractedFact],
    phase: str = "triage",
    api_key: Optional[str] = None,
    use_cross_encoder: bool = False,
) -> Tuple[float, str]:
    """
    Main entry point for fact-to-lane matching.
    
    Automatically chooses between:
    - Hybrid matching (if lane has indices)
    - Legacy exact matching (fallback)
    """
    return compute_lane_fact_delta_hybrid(
        lane, facts, phase, api_key, use_cross_encoder
    )


def compute_retrieval_prior_deltas(
    candidate_lane_ids: List[str],
    retrieval_priors: Dict[str, Any],
) -> List[BeliefDeltaProposal]:
    """
    Turn retrieval priors into belief deltas.
    
    TIERED SCALING FOR SEMANTIC EMBEDDINGS:
    - High-quality embeddings (0.75+) get exponentially scaled boosts
    - Creates strong initial separation to guide convergence
    """
    out: List[BeliefDeltaProposal] = []
    if not retrieval_priors:
        return out

    vals: Dict[str, float] = {}
    for lid in candidate_lane_ids:
        v = retrieval_priors.get(lid)
        try:
            vals[lid] = float(v)
        except Exception:
            vals[lid] = 0.0
    m = max(vals.values()) if vals else 0.0
    if m <= 1e-9:
        return out

    for lid in candidate_lane_ids:
        raw_score = vals.get(lid, 0.0)
        
        # Normalize to [0, 1] relative to best match
        normalized = raw_score / m
        
        # TIERED EXPONENTIAL SCALING
        if normalized >= 0.95:
            delta = 0.45
        elif normalized >= 0.90:
            delta = 0.35
        elif normalized >= 0.85:
            delta = 0.25
        elif normalized >= 0.75:
            delta = 0.15
        elif normalized >= 0.65:
            delta = 0.08
        elif normalized >= 0.50:
            delta = 0.0
        else:
            delta = -0.08
        
        out.append(BeliefDeltaProposal(
            key=lid, 
            delta=delta, 
            reason=f"retrieval_prior {raw_score:.4f} (normalized={normalized:.3f})"
        ))
    
    return out


# =============================================================================
# Step evaluation with validations and transitions
# =============================================================================

def evaluate_current_step(
    *,
    lane: ProcedureLane,
    step: ProcedureStep,
    facts: Dict[str, ExtractedFact],
    slots: Dict[str, Any],
    artifacts_flags: Dict[str, Any],
    ctx: Dict[str, Any],
    belief_signals: Dict[str, Any],
) -> StepEvalResult:
    """
    Evaluate validations + pass/fail constraints + transitions deterministically.
    """
    res = StepEvalResult()

    lane_delta = 0.0
    for vr in (step.validations or [])[:50]:
        hit = False
        try:
            hit = eval_constraints(vr.when, facts, slots, artifacts_flags, ctx, belief_signals)
        except Exception:
            hit = False
        res.validation_hits.append((vr.rule_id, bool(hit)))
        if not hit:
            continue

        if vr.effect == "increase_belief":
            lane_delta += float(vr.weight) * 0.07
        elif vr.effect == "decrease_belief":
            lane_delta -= float(vr.weight) * 0.07
        elif vr.effect == "terminal_resolved":
            res.terminal = "resolved"
        elif vr.effect == "terminal_escalate":
            res.terminal = "escalate"

    lane_delta = max(-1.0, min(1.0, lane_delta))
    res.belief_delta = lane_delta
    if abs(lane_delta) > 1e-6:
        res.belief_reason = "step_validations"

    # pass/fail
    if not _constraints_empty(step.pass_when):
        try:
            res.passed = bool(eval_constraints(step.pass_when, facts, slots, artifacts_flags, ctx, belief_signals))
        except Exception:
            res.passed = False

    if not _constraints_empty(step.fail_when):
        try:
            res.failed = bool(eval_constraints(step.fail_when, facts, slots, artifacts_flags, ctx, belief_signals))
        except Exception:
            res.failed = False

    # transitions
    def pick_edge(edges: List) -> Optional[str]:
        for e in edges or []:
            try:
                if _constraints_empty(e.when) or eval_constraints(e.when, facts, slots, artifacts_flags, ctx, belief_signals):
                    return e.to_step_id
            except Exception:
                continue
        return None

    if res.failed is True and step.on_fail:
        res.transition_to = pick_edge(step.on_fail)
    elif res.passed is True and step.on_pass:
        res.transition_to = pick_edge(step.on_pass)

    return res


# =============================================================================
# LLM client wrapper
# =============================================================================

@dataclass
class LLMConfig:
    model: str
    api_key: str
    base_url: Optional[str] = None
    timeout_s: float = 30.0
    max_retries: int = 3


class LLMClient:
    """
    Minimal async OpenAI-compatible client wrapper.
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import AsyncOpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("openai package not installed or not available") from e

        kwargs: Dict[str, Any] = {"api_key": self.cfg.api_key}
        if self.cfg.base_url:
            kwargs["base_url"] = self.cfg.base_url
        self._client = AsyncOpenAI(**kwargs)

    async def json_chat(self, system: str, user: str, *, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Returns parsed JSON dict. Retries on transient errors and JSON parse failures.
        """
        self._ensure_client()

        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.2 * attempt)
        raise RuntimeError(f"LLM json_chat failed after retries: {last_err}") from last_err


# =============================================================================
# Prompt builders
# =============================================================================

def build_extractor_prompt(
    *,
    fact_specs: List[FactSpec],
    user_text: str,
    artifacts: List[Dict[str, Any]],
    turn_id: str,
    ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Extractor sees allowed facts + user input + optional ctx["history"].
    
    ENHANCED EXTRACTION:
    - More aggressive multi-fact extraction from verbose responses
    - Clear examples of extracting multiple facts from compound sentences
    """
    system = (
        "You are a strict information extractor for an IT support agent.\n"
        "You MUST output valid JSON ONLY (no markdown).\n\n"
        "CRITICAL RULES:\n"
        "1. Extract facts ONLY using the allowed fact keys provided\n"
        "2. Extract ALL facts that are EXPLICITLY mentioned in the user's current message\n"
        "3. Be THOROUGH when extracting - if user says multiple things, extract ALL of them:\n"
        "   - Example: 'phone is charged, powered on with SIM' → extract device_charged=true, device_powered_on=true, sim_inserted=true\n"
        "   - Example: 'MDM is installed and initial setup is complete' → extract mdm_installed=true, setup_complete=true\n"
        "4. Use conversation_context to DISAMBIGUATE facts, NOT to extract old facts\n"
        "5. If user says 'yes' or 'no', look at conversation_context to determine WHICH fact they're confirming\n"
        "6. Do NOT invent or hallucinate facts that aren't in the current user_input\n"
        "7. If a fact is not clearly present in THIS message, omit it completely\n"
        "8. Provide a confidence in [0,1] for each extracted fact\n"
        "9. Do not include sensitive secrets (passwords, OTPs)\n\n"
        "Example disambiguation:\n"
        "- If conversation_context mentions 'Is VPN installed?' and user says 'yes', extract vpn_installed=true\n"
        "- If user just says 'yes' with no context, DO NOT guess which fact to extract\n\n"
        "Example thorough extraction:\n"
        "- User: 'yes the phone is charged and powered on with SIM'\n"
        "- Extract: device_charged=true (confidence 0.95), device_powered_on=true (confidence 0.95), sim_inserted=true (confidence 0.95)\n"
    )

    allowed = [
        {
            "fact": fs.fact,
            "type": fs.type,
            "allowed_values": fs.allowed_values,
            "description": fs.description,
            "extraction_hint": fs.extraction_hint,
        }
        for fs in fact_specs
    ]

    history = ""
    if ctx and isinstance(ctx.get("history"), str) and ctx.get("history").strip():
        history = ctx["history"].strip()

    payload = {
        "turn_id": turn_id,
        "allowed_facts": allowed,
        "conversation_context": history,
        "user_input": user_text,
        "artifacts": artifacts,
        "output_schema": {
            "extraction": {
                "received": [{"evidence_id": "string", "type": "text|screenshot|logs|file|recording|audio", "summary": "string", "sha256": "string|null", "meta": {}}],
                "extracted_facts": [{"fact": "string", "value": "any", "confidence": 0.0, "source": "chat|voice|artifact|system", "evidence_id": "string|null"}],
                "missing_fact_keys": ["string"],
                "notes": ["string"],
            }
        },
        "rules": [
            "Use evidence_id 'ev:<turn_id>:text' for the user_input.",
            "If artifacts exist, create evidence items for them too (ev:<turn_id>:a1, ev:<turn_id>:a2...).",
            "Keep summaries short; do not paste full logs.",
            "Extract ALL facts mentioned in the user's message - be thorough, not conservative.",
        ],
    }

    user = json.dumps(payload, ensure_ascii=False)
    return system, user


def build_controller_prompt(
    *,
    controller_state: ControllerState,
    candidate_lanes: List[ProcedureLane],
    extracted: ExtractionTrace,
    turn_id: str,
    ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    system = (
        "You are a controller assistant for an IT support agent.\n"
        "You MUST output valid JSON ONLY (no markdown).\n"
        "You are given:\n"
        "- current controller snapshot (belief, budgets, lane/step pointers)\n"
        "- extracted facts from the latest user turn\n"
        "- candidate procedure lanes (docs) with steps and evidence actions\n\n"
        "Your job:\n"
        "1) Propose belief_deltas over lane_ids based on extracted facts.\n"
        "2) Propose the next action: ASK_EVIDENCE, EXEC_STEP, SWITCH_LANE, ESCALATE, or RESOLVE.\n\n"
        "CRITICAL RULES:\n"
        "- Treat current belief as authoritative; your deltas should be SMALL nudges (±0.05 to ±0.15 typical)\n"
        "- If current_lane_id is set and belief > 0.5, strongly prefer staying in that lane unless facts clearly contradict it\n"
        "- Prefer ASK_EVIDENCE if no lane is clearly confident (belief < 0.6)\n"
        "- Prefer EXEC_STEP if a lane is above threshold (0.6+) or budget requires safe commit\n"
        "- SWITCH_LANE should ONLY be proposed if:\n"
        "  a) Alternative lane belief is much higher (0.75+) AND\n"
        "  b) Current lane belief has dropped significantly (<0.4) AND\n"
        "  c) The extracted facts clearly don't match current lane's purpose\n"
        "- Never propose repeating an evidence_action_id already asked\n"
        "- Keep rationale one concise paragraph\n"
        "- When in doubt, ASK_EVIDENCE to gather more information\n"
    )

    def lane_summary(l: ProcedureLane) -> Dict[str, Any]:
        step_map = l.step_map()
        entry = step_map.get(l.entry_step_id)
        entry_evidence = []
        if entry:
            entry_evidence = [
                {
                    "action_id": ea.action_id,
                    "kind": ea.kind,
                    "intent": ea.intent,
                    "prompt": ea.request.prompt,
                    "targets": ea.request.targets,
                    "cost": ea.cost,
                    "expected_info_gain": ea.expected_info_gain,
                }
                for ea in entry.evidence_actions[:6]
            ]

        cur_step = None
        if controller_state.current_lane_id == l.lane_id and controller_state.current_step_id:
            cur_step_obj = step_map.get(controller_state.current_step_id)
            if cur_step_obj:
                cur_step = {
                    "step_id": cur_step_obj.step_id,
                    "title": cur_step_obj.title,
                    "instruction": cur_step_obj.instruction,
                    "evidence_actions": [
                        {
                            "action_id": ea.action_id,
                            "kind": ea.kind,
                            "intent": ea.intent,
                            "prompt": ea.request.prompt,
                            "targets": ea.request.targets,
                            "cost": ea.cost,
                            "expected_info_gain": ea.expected_info_gain,
                        }
                        for ea in cur_step_obj.evidence_actions[:6]
                    ],
                    "validations_count": len(cur_step_obj.validations or []),
                }

        return {
            "lane_id": l.lane_id,
            "title": l.title,
            "summary": l.summary,
            "entry_step_id": l.entry_step_id,
            "preconditions": [{"slot": pc.slot, "values": pc.values} for pc in (l.preconditions or [])[:12]],
            "signatures": (l.signatures or [])[:12],
            "fact_keys": [fs.fact for fs in l.fact_specs][:40],
            "ask_slots": (l.ask_slots or [])[:20],
            "entry_step": {
                "step_id": entry.step_id if entry else l.entry_step_id,
                "title": entry.title if entry else None,
                "instruction": entry.instruction if entry else None,
                "evidence_actions": entry_evidence,
            },
            "current_step": cur_step,
        }

    retrieval_priors = {}
    if ctx and isinstance(ctx.get("retrieval_priors"), dict):
        retrieval_priors = ctx.get("retrieval_priors") or {}

    snapshot = {
        "turn_id": turn_id,
        "controller": {
            "phase": controller_state.phase.value,
            "current_lane_id": controller_state.current_lane_id,
            "current_step_id": controller_state.current_step_id,
            "belief_topk": [bi.model_dump() for bi in controller_state.belief.topk(6)],
            "budgets": controller_state.budgets.model_dump(),
            "switching": controller_state.switching.model_dump(),
            "counters": controller_state.counters.model_dump(),
            "asked_evidence_action_ids_tail": controller_state.asked_evidence_action_ids[-30:],
            "known_facts": [
                {"fact": k, "value": v.value, "confidence": v.confidence, "source": v.source, "evidence_id": v.evidence_id}
                for k, v in list(controller_state.known_facts.items())[:60]
            ],
        },
        "retrieval_priors": retrieval_priors,
        "extracted": extracted.model_dump(),
        "candidate_lanes": [lane_summary(l) for l in candidate_lanes],
        "output_schema": {
            "belief_deltas": [{"key": "lane_id", "delta": 0.0, "reason": "string"}],
            "next_action": {
                "type": "ASK_EVIDENCE|EXEC_STEP|SWITCH_LANE|ESCALATE|RESOLVE",
                "reason_code": "one_of_policy_reason_codes",
                "rationale": "string",
                "lane_id": "string|null",
                "step_id": "string|null",
                "evidence_action_id": "string|null",
            },
        },
        "policy_reason_codes": [
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
        ],
    }

    user = json.dumps(snapshot, ensure_ascii=False)
    return system, user


# =============================================================================
# Belief application
# =============================================================================

def apply_belief_deltas(
    *,
    state: ControllerState,
    candidate_lane_ids: List[str],
    deltas: List[BeliefDeltaProposal],
) -> Tuple[BeliefState, List[BeliefDelta]]:
    """Apply belief deltas with adaptive temperature."""
    before = state.belief.as_dict()

    scores: Dict[str, float] = {}
    for lid in candidate_lane_ids:
        p = before.get(lid)
        if p is None:
            p = 1.0 / max(1, len(candidate_lane_ids))
        scores[lid] = float(p)

    delta_map: Dict[str, List[BeliefDeltaProposal]] = {}
    for d in deltas:
        if d.key in scores:
            delta_map.setdefault(d.key, []).append(d)
            scores[d.key] = scores.get(d.key, 0.0) + float(d.delta)

    for k in list(scores.keys()):
        if scores[k] < 0.0:
            scores[k] = 0.0

    # USE ADAPTIVE SOFTMAX
    current_top_prob = max(before.values()) if before else 0.0
    probs = _softmax(
        scores, 
        temperature=state.belief.temperature,
        turn_count=state.counters.total_turns,
        phase=state.phase.value,
        current_top_prob=current_top_prob,
    )

    updated = BeliefState(
        items=[BeliefItem(key=k, belief=float(probs.get(k, 0.0))) for k in candidate_lane_ids],
        temperature=state.belief.temperature,  # Store base temp
    )

    traces: List[BeliefDelta] = []
    for lid in candidate_lane_ids:
        b = float(before.get(lid, 0.0))
        a = float(probs.get(lid, 0.0))
        reasons = delta_map.get(lid, [])
        reason_txt = "; ".join([r.reason for r in reasons][:3]) if reasons else "normalized"
        if abs(a - b) >= 1e-6:
            traces.append(BeliefDelta(key=lid, before=b, after=a, reason=reason_txt))

    return updated, traces


def compute_top_gap(belief: BeliefState, current_lane_id: Optional[str]) -> Tuple[Optional[str], float, float]:
    items = sorted(belief.items, key=lambda x: x.belief, reverse=True)
    if not items:
        return None, 0.0, 0.0
    top = items[0]
    cur = next((x for x in items if x.key == current_lane_id), None)
    gap = top.belief - (cur.belief if cur else 0.0)
    return top.key, float(top.belief), float(gap)


# =============================================================================
# Evidence action selection
# =============================================================================

def _triage_evidence_bank() -> List[EvidenceAction]:
    """
    Generic evidence actions used when lanes provide no good entry evidence_actions.
    """
    return [
        EvidenceAction(
            action_id="triage:os_device",
            kind="question",
            intent="Identify OS and device type to narrow down procedures",
            request=EvidenceRequest(
                type="text",
                prompt="Which OS and device is this on? (Windows/macOS/Linux, and laptop/desktop/mobile)",
                targets=["os", "device_type"],
                examples=[],
                redact_hints=[],
                parse_hint="Extract os and device_type",
            ),
            cost="low",
            expected_info_gain=0.6,
        ),
        EvidenceAction(
            action_id="triage:app",
            kind="question",
            intent="Identify the affected application/product",
            request=EvidenceRequest(
                type="text",
                prompt="Which app or site is affected? (e.g., Outlook/Teams/Chrome/internal portal)",
                targets=["app", "product"],
                examples=[],
                redact_hints=[],
                parse_hint="Extract app/product",
            ),
            cost="low",
            expected_info_gain=0.6,
        ),
        EvidenceAction(
            action_id="triage:error_text",
            kind="question",
            intent="Capture exact error text/codes for signature matching",
            request=EvidenceRequest(
                type="text",
                prompt="What's the exact error message or error code you see? (copy/paste if possible)",
                targets=["error_message", "error_code"],
                examples=[],
                redact_hints=[],
                parse_hint="Extract exact error_message/error_code",
            ),
            cost="low",
            expected_info_gain=0.8,
        ),
        EvidenceAction(
            action_id="triage:network",
            kind="question",
            intent="Determine network/VPN context",
            request=EvidenceRequest(
                type="text",
                prompt="Are you on VPN or office network right now? (Yes/No + home/office)",
                targets=["vpn_on", "network_location"],
                examples=[],
                redact_hints=[],
                parse_hint="Extract vpn_on and network_location",
            ),
            cost="low",
            expected_info_gain=0.5,
        ),
        EvidenceAction(
            action_id="triage:screenshot",
            kind="artifact_request",
            intent="A screenshot can disambiguate errors quickly",
            request=EvidenceRequest(
                type="screenshot",
                prompt="If you can, please share a screenshot of the error.",
                targets=["error_message", "error_code"],
                examples=[],
                redact_hints=["token", "api_key"],
                parse_hint="Extract visible error text/codes from screenshot",
            ),
            cost="med",
            expected_info_gain=0.7,
        ),
    ]


def _asked_set(state: ControllerState) -> set:
    return set(state.asked_evidence_action_ids or [])


def _recently_asked(state: ControllerState, ea_id: str, window: int = 12) -> bool:
    tail = state.asked_evidence_action_ids[-window:] if state.asked_evidence_action_ids else []
    return ea_id in tail


def _missing_facts(state: ControllerState) -> set:
    out = set()
    for k, ef in (state.known_facts or {}).items():
        if ef is not None and ef.value is not None:
            out.add(_canon_key(k))
    return out


def build_slot_question_evidence_actions(
    *,
    candidate_lanes: List[ProcedureLane],
    state: ControllerState,
    topk_order: Optional[List[str]] = None,
    max_actions: int = 10,
) -> List[EvidenceAction]:
    """
    Turn lane.ask_slots/slot_questions into EvidenceActions for TRIAGE.
    """
    have = _missing_facts(state)
    asked = _asked_set(state)

    lane_map = {l.lane_id: l for l in candidate_lanes}
    lane_ids = topk_order or [l.lane_id for l in candidate_lanes]

    out: List[EvidenceAction] = []

    for lid in lane_ids:
        lane = lane_map.get(lid)
        if not lane:
            continue

        safety_map: Dict[str, str] = {}
        if isinstance(lane.slot_safety, dict):
            safety_map = {_canon_key(k): str(v) for k, v in lane.slot_safety.items()}
        elif isinstance(lane.slot_safety, list):
            safety_map = {_canon_key(e.slot): str(e.safety) for e in lane.slot_safety}

        q_map = {_canon_key(sq.slot): (sq.question or "").strip() for sq in (lane.slot_questions or [])}

        order = []
        if lane.slot_priority:
            order = [_canon_key(x) for x in lane.slot_priority if _canon_key(x)]
        elif lane.ask_slots:
            order = [_canon_key(x) for x in lane.ask_slots if _canon_key(x)]

        for slot in order:
            if not slot:
                continue
            if safety_map.get(slot, "safe") != "safe":
                continue
            if slot in have:
                continue

            q = q_map.get(slot) or f"Can you share the {slot.replace('_',' ')}?"
            ea_id = f"slotq:{lid}:{slot}"

            if ea_id in asked or _recently_asked(state, ea_id):
                continue

            out.append(
                EvidenceAction(
                    action_id=ea_id,
                    kind="question",
                    intent=f"Gather '{slot}' to disambiguate the issue",
                    request=EvidenceRequest(
                        type="text",
                        prompt=q,
                        targets=[slot],
                        examples=[],
                        redact_hints=[],
                        parse_hint=f"Extract {slot}",
                    ),
                    cost="low",
                    expected_info_gain=0.55,
                )
            )
            if len(out) >= max_actions:
                return out

    return out


def select_next_evidence_action(
    *,
    candidate_lanes: List[ProcedureLane],
    state: ControllerState,
) -> Optional[EvidenceAction]:
    """
    IMPROVED EVIDENCE SELECTION:
    - Try top lane first (highest priority)
    - Then try top 3 lanes (was top 2)
    - Only expand to top 6 if beliefs are very flat (<0.03 spread, was 0.05)
    - Fallback to triage bank as last resort
    """
    asked = _asked_set(state)

    top_lane_id, top_prob, _ = compute_top_gap(state.belief, state.current_lane_id)
    enter_th = state.switching.enter_lane_threshold

    def pick_best(actions: List[EvidenceAction]) -> Optional[EvidenceAction]:
        filtered = []
        for a in actions or []:
            if not a or not a.action_id:
                continue
            if a.action_id in asked:
                continue
            if _recently_asked(state, a.action_id):
                continue
            filtered.append(a)

        if not filtered:
            return None

        cost_rank = {"low": 0, "med": 1, "high": 2}
        filtered.sort(key=lambda a: (-(a.expected_info_gain or 0.0), cost_rank.get(a.cost, 1)))
        return filtered[0]

    # EXECUTE phase: prefer current step evidence
    if state.phase == Phase.EXECUTE and state.current_lane_id and state.current_step_id:
        lane = next((l for l in candidate_lanes if l.lane_id == state.current_lane_id), None)
        if lane:
            step = lane.step_map().get(state.current_step_id)
            if step and step.evidence_actions:
                best = pick_best(step.evidence_actions)
                if best:
                    return best

    # TRIAGE phase: IMPROVED STRATEGY
    if state.phase == Phase.TRIAGE and top_prob < enter_th:
        # Get belief-sorted lanes
        belief_sorted = sorted(state.belief.items, key=lambda x: x.belief, reverse=True)
        
        # Try top lane first (highest priority - 80% weight)
        if belief_sorted:
            top_lane = belief_sorted[0]
            slot_eas_top = build_slot_question_evidence_actions(
                candidate_lanes=candidate_lanes,
                state=state,
                topk_order=[top_lane.key],
                max_actions=5,
            )
            best = pick_best(slot_eas_top)
            if best:
                return best
        
        # Try top 3 lanes (INCREASED from 2, lowered threshold from 0.15 to 0.12)
        if len(belief_sorted) >= 3 and belief_sorted[0].belief > 0.12:
            top_3 = [bi.key for bi in belief_sorted[:3]]
            slot_eas_top3 = build_slot_question_evidence_actions(
                candidate_lanes=candidate_lanes,
                state=state,
                topk_order=top_3,
                max_actions=8,
            )
            best = pick_best(slot_eas_top3)
            if best:
                return best
        
        # Only expand to top 6 if beliefs are VERY flat (TIGHTENED from 0.05 to 0.03)
        if belief_sorted:
            belief_spread = belief_sorted[0].belief - belief_sorted[min(3, len(belief_sorted)-1)].belief
            
            # If beliefs are very flat (<0.03 spread), use broader search
            if belief_spread < 0.03:
                top_order = [bi.key for bi in belief_sorted[:6]]
                slot_eas = build_slot_question_evidence_actions(
                    candidate_lanes=candidate_lanes,
                    state=state,
                    topk_order=top_order,
                    max_actions=12,
                )
                best = pick_best(slot_eas)
                if best:
                    return best

        best = pick_best(_triage_evidence_bank())
        if best:
            return best

    # Use entry step evidence from top lane
    if top_lane_id:
        lane = next((l for l in candidate_lanes if l.lane_id == top_lane_id), None)
        if lane:
            entry = lane.step_map().get(lane.entry_step_id)
            if entry and entry.evidence_actions:
                best = pick_best(entry.evidence_actions)
                if best:
                    return best

    # Last resort
    for lane in candidate_lanes:
        entry = lane.step_map().get(lane.entry_step_id)
        if entry and entry.evidence_actions:
            best = pick_best(entry.evidence_actions)
            if best:
                return best

    return pick_best(_triage_evidence_bank())


# =============================================================================
# Policy enforcement
# =============================================================================

def enforce_policy(
    *,
    proposal: ControllerProposal,
    state: ControllerState,
    candidate_lanes: List[ProcedureLane],
) -> Tuple[PolicyTrace, Optional[StepTrace], List[str]]:
    warnings: List[str] = []
    pol = proposal.next_action

    lane_ids = {l.lane_id for l in candidate_lanes}
    lane_map = {l.lane_id: l for l in candidate_lanes}

    top_lane_id, top_prob, _ = compute_top_gap(state.belief, state.current_lane_id)

    budgets = state.budgets
    ctr = state.counters
    switching = state.switching

    if pol.type == "RESOLVE":
        return (
            PolicyTrace(
                selected_action="RESOLVE",
                reason_code=pol.reason_code,
                rationale=pol.rationale,
                selected_lane_id=state.current_lane_id,
                selected_step_id=state.current_step_id,
            ),
            None,
            warnings,
        )

    if pol.type == "ESCALATE":
        return (
            PolicyTrace(
                selected_action="ESCALATE",
                reason_code=pol.reason_code,
                rationale=pol.rationale,
            ),
            None,
            warnings,
        )

    if pol.type == "SWITCH_LANE":
        if not pol.lane_id or pol.lane_id not in lane_ids:
            warnings.append("Proposed SWITCH_LANE has invalid lane_id; blocking.")
        else:
            alt = pol.lane_id
            alt_prob = next((x.belief for x in state.belief.items if x.key == alt), 0.0)
            cur_prob = next((x.belief for x in state.belief.items if x.key == state.current_lane_id), 0.0)

            progress_cost = compute_progress_switch_cost(ctr.steps_executed_in_lane, switching.max_progress_switch_cost)
            required_margin = switching.switch_margin + progress_cost

            if alt_prob < switching.strong_lane_threshold:
                warnings.append("SWITCH_LANE blocked: alt below strong_lane_threshold.")
                return (
                    PolicyTrace(
                        selected_action="ASK_EVIDENCE",
                        reason_code="LANE_SWITCH_BLOCKED_LOW_CONF",
                        rationale="Switch suggestion rejected (alt lane not strong enough). Asking for more evidence.",
                    ),
                    None,
                    warnings,
                )
            if (alt_prob - cur_prob) < required_margin:
                warnings.append("SWITCH_LANE blocked: margin too small.")
                return (
                    PolicyTrace(
                        selected_action="ASK_EVIDENCE",
                        reason_code="LANE_SWITCH_BLOCKED_SMALL_MARGIN",
                        rationale="Switch suggestion rejected (margin too small). Asking for more evidence.",
                    ),
                    None,
                    warnings,
                )

            if state.switch_vote_lane_id == alt:
                state.switch_vote_count += 1
            else:
                state.switch_vote_lane_id = alt
                state.switch_vote_count = 1

            if state.switch_vote_count < switching.require_consecutive_turns:
                warnings.append("SWITCH_LANE pending: needs more consecutive votes.")
                return (
                    PolicyTrace(
                        selected_action="ASK_EVIDENCE",
                        reason_code="LANE_SWITCH_BLOCKED_SMALL_MARGIN",
                        rationale="Switch looks plausible but waiting for confirmation next turn; asking clarifying evidence.",
                    ),
                    None,
                    warnings,
                )

            return (
                PolicyTrace(
                    selected_action="SWITCH_LANE",
                    reason_code=pol.reason_code,
                    rationale=pol.rationale,
                    selected_lane_id=alt,
                    selected_step_id=lane_map[alt].entry_step_id,
                ),
                None,
                warnings,
            )

    if pol.type == "EXEC_STEP":
        lane_id = pol.lane_id or state.current_lane_id or top_lane_id
        if not lane_id or lane_id not in lane_ids:
            warnings.append("EXEC_STEP invalid lane_id; falling back to ASK_EVIDENCE.")
            ea = select_next_evidence_action(candidate_lanes=candidate_lanes, state=state)
            return (
                PolicyTrace(
                    selected_action="ASK_EVIDENCE",
                    reason_code="THRESHOLD_NOT_MET",
                    rationale="Cannot execute step due to invalid lane; asking for evidence.",
                    selected_evidence_action_id=ea.action_id if ea else None,
                ),
                None,
                warnings,
            )

        lane = lane_map[lane_id]
        sm = lane.step_map()

        step_id = pol.step_id or state.current_step_id or lane.entry_step_id
        if step_id not in sm:
            warnings.append("EXEC_STEP invalid step_id; using entry_step_id.")
            step_id = lane.entry_step_id

        if top_prob < switching.enter_lane_threshold:
            if ctr.precommit_questions < budgets.max_precommit_questions:
                ea = select_next_evidence_action(candidate_lanes=candidate_lanes, state=state)
                return (
                    PolicyTrace(
                        selected_action="ASK_EVIDENCE",
                        reason_code="THRESHOLD_NOT_MET",
                        rationale="Not confident enough to execute; asking high-info evidence first.",
                        selected_evidence_action_id=ea.action_id if ea else None,
                    ),
                    None,
                    warnings,
                )

        st = StepTrace(
            lane_id=lane_id,
            step_id=step_id,
            action_taken="EXEC_STEP",
            asked_evidence_action_ids=[],
            validation_results=[],
            transition_taken=None,
        )
        return (
            PolicyTrace(
                selected_action="EXEC_STEP",
                reason_code=pol.reason_code,
                rationale=pol.rationale,
                selected_lane_id=lane_id,
                selected_step_id=step_id,
            ),
            st,
            warnings,
        )

    if pol.type == "ASK_EVIDENCE":
        if ctr.total_questions >= budgets.max_total_questions:
            return (
                PolicyTrace(
                    selected_action="ESCALATE",
                    reason_code="BUDGET_REACHED_ESCALATE",
                    rationale="Question budget exhausted; escalating with required evidence request.",
                ),
                None,
                warnings,
            )

        ea: Optional[EvidenceAction] = None
        asked = set(state.asked_evidence_action_ids or [])

        if pol.evidence_action_id and pol.evidence_action_id not in asked and not _recently_asked(state, pol.evidence_action_id):
            for lane in candidate_lanes:
                for step in lane.steps:
                    for a in step.evidence_actions:
                        if a.action_id == pol.evidence_action_id:
                            ea = a
                            break

        if ea is None:
            ea = select_next_evidence_action(candidate_lanes=candidate_lanes, state=state)

        return (
            PolicyTrace(
                selected_action="ASK_EVIDENCE",
                reason_code=pol.reason_code,
                rationale=pol.rationale,
                selected_evidence_action_id=ea.action_id if ea else None,
            ),
            None,
            warnings,
        )

    warnings.append("Unknown policy branch; default ASK_EVIDENCE.")
    ea = select_next_evidence_action(candidate_lanes=candidate_lanes, state=state)
    return (
        PolicyTrace(
            selected_action="ASK_EVIDENCE",
            reason_code="THRESHOLD_NOT_MET",
            rationale="Fallback to evidence collection.",
            selected_evidence_action_id=ea.action_id if ea else None,
        ),
        None,
        warnings,
    )


# =============================================================================
# Main orchestrator: one turn
# =============================================================================

async def run_turn(
    *,
    llm_extractor: LLMClient,
    llm_controller: LLMClient,
    controller_state: ControllerState,
    candidate_lanes: List[ProcedureLane],
    user_text: str,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    slots: Optional[Dict[str, Any]] = None,
    ctx: Optional[Dict[str, Any]] = None,
) -> AgentTurnResult:
    """
    One "turn" processes the latest user input and returns:
    - assistant response text
    - updated ControllerState
    - TurnTrace
    """
    artifacts = artifacts or []
    slots = slots or {}
    ctx = ctx or {}
    
    # Track warnings throughout the turn
    warnings: List[str] = []

    controller_state.counters.total_turns += 1
    turn_id = f"turn:{controller_state.counters.total_turns}"

    candidate_lane_ids = [l.lane_id for l in candidate_lanes]

    # Seed belief from retrieval priors
    retrieval_priors = ctx.get("retrieval_priors") if isinstance(ctx.get("retrieval_priors"), dict) else {}
    deterministic_delta_proposals: List[BeliefDeltaProposal] = []

    deterministic_delta_proposals.extend(
        compute_retrieval_prior_deltas(candidate_lane_ids=candidate_lane_ids, retrieval_priors=retrieval_priors or {})
    )

    # LLM #1: extractor
    fact_spec_map: Dict[str, FactSpec] = {}
    for lane in candidate_lanes:
        for fs in lane.fact_specs:
            fact_spec_map.setdefault(_canon_key(fs.fact), fs)
    fact_specs = list(fact_spec_map.values())

    sys1, usr1 = build_extractor_prompt(
        fact_specs=fact_specs,
        user_text=user_text,
        artifacts=artifacts,
        turn_id=turn_id,
        ctx=ctx,
    )

    raw1 = await llm_extractor.json_chat(sys1, usr1, temperature=0.1)

    try:
        extractor_out = ExtractorOutput.model_validate(raw1)
    except ValidationError as ve:
        ev_id = f"ev:{turn_id}:text"
        extractor_out = ExtractorOutput(
            extraction=ExtractionTrace(
                received=[EvidenceItem(evidence_id=ev_id, type="text", summary=user_text[:200], sha256=_sha256_text(user_text))],
                extracted_facts=[],
                missing_fact_keys=[],
                notes=[f"Extractor JSON validation failed: {ve}"],
            )
        )

    extraction = extractor_out.extraction

    for ef in extraction.extracted_facts:
        ef.fact = _canon_key(ef.fact)

    controller_state.known_facts = merge_facts(controller_state.known_facts, extraction.extracted_facts)

    # Deterministic step evaluation
    step_trace: Optional[StepTrace] = None
    step_eval: Optional[StepEvalResult] = None

    artifacts_flags: Dict[str, Any] = {
        "has_screenshot": any(a.get("type") == "screenshot" for a in artifacts),
        "has_logs": any(a.get("type") == "logs" for a in artifacts),
        "has_file": any(a.get("type") == "file" for a in artifacts),
        "has_recording": any(a.get("type") in ("recording", "audio") for a in artifacts),
    }
    belief_signals: Dict[str, Any] = {
        "top_prob": None,
        "phase": controller_state.phase.value,
    }

    if controller_state.current_lane_id and controller_state.current_step_id:
        lane = next((l for l in candidate_lanes if l.lane_id == controller_state.current_lane_id), None)
        if lane:
            sm = lane.step_map()
            cur_step = sm.get(controller_state.current_step_id)
            if cur_step:
                step_eval = evaluate_current_step(
                    lane=lane,
                    step=cur_step,
                    facts=controller_state.known_facts,
                    slots=slots,
                    artifacts_flags=artifacts_flags,
                    ctx=ctx,
                    belief_signals=belief_signals,
                )

                if abs(step_eval.belief_delta) > 1e-9:
                    deterministic_delta_proposals.append(
                        BeliefDeltaProposal(
                            key=lane.lane_id,
                            delta=max(-1.0, min(1.0, step_eval.belief_delta)),
                            reason=step_eval.belief_reason or "step_validations",
                        )
                    )

                if step_eval.failed is True:
                    controller_state.counters.failed_steps += 1

                if step_eval.terminal == "resolved":
                    policy = PolicyTrace(
                        selected_action="RESOLVE",
                        reason_code="TERMINAL_VALIDATION_RESOLVED",
                        rationale="Terminal validation indicates the issue is resolved.",
                        selected_lane_id=lane.lane_id,
                        selected_step_id=cur_step.step_id,
                    )
                    trace = TurnTrace(
                        turn_id=turn_id,
                        extraction=extraction,
                        belief=BeliefTrace(phase=controller_state.phase.value),
                        policy=policy,
                        step=StepTrace(
                            lane_id=lane.lane_id,
                            step_id=cur_step.step_id,
                            action_taken="RESOLVE",
                            asked_evidence_action_ids=[],
                            validation_results=step_eval.validation_hits,
                            transition_taken=None,
                            notes=["terminal_resolved"],
                        ),
                        warnings=[],
                    )
                    controller_state.phase = Phase.TRIAGE
                    controller_state.current_lane_id = None
                    controller_state.current_step_id = None
                    return AgentTurnResult(
                        response_text="Great — sounds like it's resolved. If it happens again, share the exact error text and when it started.",
                        state=controller_state,
                        trace=trace,
                    )

                if step_eval.terminal == "escalate":
                    policy = PolicyTrace(
                        selected_action="ESCALATE",
                        reason_code="TERMINAL_VALIDATION_ESCALATE",
                        rationale="Terminal validation indicates escalation is required.",
                        selected_lane_id=lane.lane_id,
                        selected_step_id=cur_step.step_id,
                    )
                    trace = TurnTrace(
                        turn_id=turn_id,
                        extraction=extraction,
                        belief=BeliefTrace(phase=controller_state.phase.value),
                        policy=policy,
                        step=StepTrace(
                            lane_id=lane.lane_id,
                            step_id=cur_step.step_id,
                            action_taken="ESCALATE",
                            asked_evidence_action_ids=[],
                            validation_results=step_eval.validation_hits,
                            transition_taken=None,
                            notes=["terminal_escalate"],
                        ),
                        warnings=[],
                    )
                    controller_state.phase = Phase.TRIAGE
                    controller_state.current_lane_id = None
                    controller_state.current_step_id = None
                    return AgentTurnResult(
                        response_text=(
                            "I can't proceed safely without escalation based on what we observed. "
                            "Please share: exact error text, OS/app version, when it started, and a screenshot/logs if available."
                        ),
                        state=controller_state,
                        trace=trace,
                    )

                if step_eval.transition_to and step_eval.transition_to in sm:
                    controller_state.current_step_id = step_eval.transition_to
                    controller_state.phase = Phase.EXECUTE

                    step_trace = StepTrace(
                        lane_id=lane.lane_id,
                        step_id=cur_step.step_id,
                        action_taken="EXEC_STEP",
                        asked_evidence_action_ids=[],
                        validation_results=step_eval.validation_hits,
                        transition_taken=step_eval.transition_to,
                        notes=["transition_taken"],
                    )

    # === HYBRID FACT MATCHING (NEW) ===
    # Determine if cross-encoder should be used (high-stakes decisions only)
    use_cross_encoder = (
        controller_state.counters.total_questions >= 2 and
        controller_state.phase == Phase.TRIAGE and
        len(controller_state.known_facts) >= 2
    )
    
    # Get API key for embedding computation
    api_key = llm_extractor.cfg.api_key
    
    # Compute hybrid fact-to-lane deltas
    for lane in candidate_lanes:
        d, reason = compute_lane_fact_delta(
            lane,
            controller_state.known_facts,
            phase=controller_state.phase.value,
            api_key=api_key,
            use_cross_encoder=use_cross_encoder,
        )
        if abs(d) > 1e-6:
            deterministic_delta_proposals.append(
                BeliefDeltaProposal(key=lane.lane_id, delta=d, reason=reason)
            )

    # LLM #2: controller proposal
    sys2, usr2 = build_controller_prompt(
        controller_state=controller_state,
        candidate_lanes=candidate_lanes,
        extracted=extraction,
        turn_id=turn_id,
        ctx=ctx,
    )

    raw2 = await llm_controller.json_chat(sys2, usr2, temperature=0.2)

    try:
        proposal = ControllerProposal.model_validate(raw2)
    except ValidationError as ve:
        proposal = ControllerProposal(
            belief_deltas=[],
            next_action=NextActionProposal(
                type="ASK_EVIDENCE",
                reason_code="THRESHOLD_NOT_MET",
                rationale=f"Controller proposal invalid; asking for more evidence. ({ve})",
            ),
        )

    # Apply belief deltas
    belief_before = controller_state.belief.topk(6)

    all_deltas: List[BeliefDeltaProposal] = []
    all_deltas.extend(deterministic_delta_proposals[:80])
    all_deltas.extend(proposal.belief_deltas[:40])

    updated_belief, belief_deltas_trace = apply_belief_deltas(
        state=controller_state,
        candidate_lane_ids=candidate_lane_ids,
        deltas=all_deltas,
    )
    controller_state.belief = updated_belief

    top_lane_id, top_prob, gap = compute_top_gap(controller_state.belief, controller_state.current_lane_id)
    
    # ADAPTIVE THRESHOLD: Decreases over time
    turns = controller_state.counters.total_turns
    num_facts = len([f for f in controller_state.known_facts.values() if f.value is not None])
    
    # Base threshold starts at 0.60, decreases by 0.04 per turn (min 0.45)
    adaptive_threshold = max(0.45, 0.60 - (min(turns - 1, 4) * 0.04))
    
    threshold_crossed = top_prob >= adaptive_threshold
    belief_signals["top_prob"] = top_prob

    # RELAXED PHASE TRANSITION
    if controller_state.current_lane_id is None:
        # Multiple ways to commit (OR logic):
        can_commit = (
            # Standard path: threshold crossed with minimal validation
            (threshold_crossed and turns >= 2) or
            
            # High confidence override (no turn requirement)
            (top_prob >= 0.70) or
            
            # Extended exploration override
            (turns >= 5 and top_prob >= 0.50) or
            
            # Strong gap indicates clear winner
            (gap >= 0.20 and turns >= 3 and num_facts >= 2)
        )
        
        if can_commit:
            controller_state.phase = Phase.EXECUTE
            warnings.append(
                f"✅ COMMITTED: prob={top_prob:.3f}, threshold={adaptive_threshold:.3f}, "
                f"turns={turns}, gap={gap:.3f}"
            )
        else:
            controller_state.phase = Phase.TRIAGE
            if turns >= 3:
                warnings.append(
                    f"⚠️ EXPLORING: prob={top_prob:.3f} (need {adaptive_threshold:.3f}), "
                    f"gap={gap:.3f}, turns={turns}, facts={num_facts}"
                )
    elif threshold_crossed:
        controller_state.phase = Phase.EXECUTE

    belief_trace = BeliefTrace(
        topk_before=belief_before,
        topk_after=controller_state.belief.topk(6),
        deltas=belief_deltas_trace,
        threshold_crossed=threshold_crossed,
        threshold_value=controller_state.switching.enter_lane_threshold,
        current_lane_id=controller_state.current_lane_id,
        candidate_lane_id=top_lane_id,
        belief_gap=gap,
        consecutive_switch_votes=controller_state.switch_vote_count,
        phase=controller_state.phase.value,
    )

    # Enforce policy
    policy_trace, proposed_step_trace, policy_warnings = enforce_policy(
        proposal=proposal,
        state=controller_state,
        candidate_lanes=candidate_lanes,
    )
    
    warnings.extend(policy_warnings)

    if step_trace is None:
        step_trace = proposed_step_trace

    # Update state counters
    if policy_trace.selected_action == "ASK_EVIDENCE":
        controller_state.counters.total_questions += 1
        if controller_state.current_lane_id is None and not threshold_crossed:
            controller_state.counters.precommit_questions += 1
        if policy_trace.selected_evidence_action_id:
            controller_state.asked_evidence_action_ids.append(policy_trace.selected_evidence_action_id)

    elif policy_trace.selected_action == "EXEC_STEP":
        controller_state.current_lane_id = policy_trace.selected_lane_id
        controller_state.current_step_id = policy_trace.selected_step_id
        controller_state.phase = Phase.EXECUTE
        controller_state.counters.steps_executed_in_lane += 1

    elif policy_trace.selected_action == "SWITCH_LANE":
        controller_state.current_lane_id = policy_trace.selected_lane_id
        controller_state.current_step_id = policy_trace.selected_step_id
        controller_state.phase = Phase.EXECUTE
        controller_state.counters.steps_executed_in_lane = 0

    elif policy_trace.selected_action in ("ESCALATE", "RESOLVE"):
        controller_state.phase = Phase.TRIAGE
        controller_state.current_lane_id = None
        controller_state.current_step_id = None

    # Render assistant response text
    response_text = ""

    def _find_evidence_action_by_id(ea_id: str) -> Optional[EvidenceAction]:
        for lane in candidate_lanes:
            for step in lane.steps:
                for ea in step.evidence_actions:
                    if ea.action_id == ea_id:
                        return ea
        for ea in _triage_evidence_bank():
            if ea.action_id == ea_id:
                return ea
        return None

    if policy_trace.selected_action == "ASK_EVIDENCE":
        prompt = None
        if policy_trace.selected_evidence_action_id:
            ea = _find_evidence_action_by_id(policy_trace.selected_evidence_action_id)
            if ea:
                prompt = ea.request.prompt
        response_text = prompt or "Can you share a bit more detail (exact error text, when it started, and what changed recently)?"

    elif policy_trace.selected_action == "EXEC_STEP":
        lane = next((l for l in candidate_lanes if l.lane_id == policy_trace.selected_lane_id), None)
        if lane:
            step = lane.step_map().get(policy_trace.selected_step_id or "")
            if step:
                response_text = step.instruction
                if step.how_to_reach:
                    response_text += "\n\nHow to reach it:\n- " + "\n- ".join(step.how_to_reach[:6])

                if step.evidence_actions:
                    asked = set(controller_state.asked_evidence_action_ids)
                    best = next((ea for ea in step.evidence_actions if ea.action_id not in asked), None)
                    if best:
                        response_text += "\n\nAfter you do that: " + best.request.prompt
                        controller_state.asked_evidence_action_ids.append(best.action_id)
                        controller_state.counters.total_questions += 1

        if not response_text:
            response_text = "Please follow the next troubleshooting step and tell me what happens."

    elif policy_trace.selected_action == "SWITCH_LANE":
        response_text = "Based on what you said, a different fix path fits better. Let's switch approaches.\n"
        lane = next((l for l in candidate_lanes if l.lane_id == policy_trace.selected_lane_id), None)
        if lane:
            step = lane.step_map().get(lane.entry_step_id)
            if step:
                response_text += "\n" + step.instruction
                if step.evidence_actions:
                    asked = set(controller_state.asked_evidence_action_ids)
                    best = next((ea for ea in step.evidence_actions if ea.action_id not in asked), None)
                    if best:
                        response_text += "\n\n" + best.request.prompt
                        controller_state.asked_evidence_action_ids.append(best.action_id)
                        controller_state.counters.total_questions += 1

    elif policy_trace.selected_action == "ESCALATE":
        response_text = (
            "I'm not confident enough to proceed without more evidence. "
            "Please share: exact error text, timestamp when it started, your OS/app version, and a screenshot or logs if available."
        )

    elif policy_trace.selected_action == "RESOLVE":
        response_text = "Great — sounds like it's resolved. If it happens again, tell me the exact error text and when it started."

    trace = TurnTrace(
        turn_id=turn_id,
        extraction=extraction,
        belief=belief_trace,
        policy=policy_trace,
        step=step_trace,
        warnings=warnings,
    )

    return AgentTurnResult(response_text=response_text, state=controller_state, trace=trace)


# =============================================================================
# Example usage (remove in production)
# =============================================================================

async def _demo():
    from pydantic_models import (
        Comparator,
        Condition,
        ConstraintSet,
        EvidenceAction,
        EvidenceRequest,
        FactSpec,
        ProcedureLane,
        ProcedureStep,
        ValidationRule,
    )

    lane = ProcedureLane(
        procedure_id="lane:vpn_basic",
        doc_id="kb-001",
        doc_title="VPN connectivity basics",
        title="Fix VPN connectivity",
        summary="Use when user cannot access internal resources; often VPN is disconnected or misconfigured.",
        fact_specs=[
            FactSpec(fact="vpn_on", description="Whether user is on VPN", type="boolean", required_for_certainty=True),
            FactSpec(fact="error_message", description="Exact error text", type="string"),
            FactSpec(fact="user_says_resolved", description="User says issue is resolved", type="boolean", required_for_certainty=True),
        ],
        signatures=["vpn", "not connected", "cannot reach internal"],
        entry_step_id="s1",
        steps=[
            ProcedureStep(
                step_id="s1",
                title="Check VPN connection",
                instruction="Please connect to the company VPN and then retry the action.",
                evidence_actions=[
                    EvidenceAction(
                        action_id="evq:vpn_status",
                        kind="question",
                        intent="Confirm VPN state",
                        request=EvidenceRequest(type="text", prompt="Is your VPN connected now? (Yes/No)", targets=["vpn_on"]),
                        expected_info_gain=0.8,
                        cost="low",
                    ),
                    EvidenceAction(
                        action_id="evq:resolved",
                        kind="confirmation",
                        intent="Check resolution",
                        request=EvidenceRequest(type="text", prompt="Did that fix it? (Yes/No)", targets=["user_says_resolved"]),
                        expected_info_gain=0.6,
                        cost="low",
                    ),
                ],
                validations=[
                    ValidationRule(
                        rule_id="v1_vpn_on",
                        description="VPN connected increases lane belief",
                        when=ConstraintSet(all_of=[Condition(lhs="fact.vpn_on", op=Comparator.EQ, rhs=True)]),
                        effect="increase_belief",
                        weight=2.0,
                    ),
                    ValidationRule(
                        rule_id="v2_resolved",
                        description="User says resolved => terminal resolved",
                        when=ConstraintSet(all_of=[Condition(lhs="fact.user_says_resolved", op=Comparator.EQ, rhs=True)]),
                        effect="terminal_resolved",
                        weight=5.0,
                    ),
                ],
            )
        ],
    )

    extractor = LLMClient(LLMConfig(model="gpt-4o-mini", api_key="YOUR_KEY"))
    controller = LLMClient(LLMConfig(model="gpt-4o-mini", api_key="YOUR_KEY"))

    state = ControllerState()
    res = await run_turn(
        llm_extractor=extractor,
        llm_controller=controller,
        controller_state=state,
        candidate_lanes=[lane],
        user_text="My internal website isn't loading and I think VPN is off",
        artifacts=[],
        ctx={"retrieval_priors": {"lane:vpn_basic": 0.9}},
    )
    print(res.response_text)
    print(json.dumps(res.trace.model_dump(), indent=2))


if __name__ == "__main__":
    # asyncio.run(_demo())
    pass