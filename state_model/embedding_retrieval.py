#!/usr/bin/env python3
"""
embedding_retrieval.py

Semantic embedding-based lane retrieval using hybrid indices from compiled procedures.

Features:
- Uses pre-computed embeddings from procedure compilation (fact_embeddings)
- Computes query embeddings on-the-fly
- Cosine similarity matching for semantic search
- Falls back to keyword matching if embeddings unavailable
- Maintains stability by boosting lanes with existing belief

Usage:
    from embedding_retrieval import EmbeddingLaneStore
    
    store = EmbeddingLaneStore.load(
        lanes_path="./out/procedures_hybrid.jsonl",
        embedding_model="text-embedding-3-small",
        use_openai=True,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    lanes, priors = store.retrieve(
        query_text="VPN connection failed",
        cfg=RetrievalConfig(top_k=5),
        controller_state=state,
    )
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import from chat_loop to use same base classes
from pydantic_models import ProcedureLane, ControllerState


@dataclass
class RetrievalConfig:
    top_k: int = 5
    min_score: float = 0.02
    prefer_current_lane: bool = True


# =============================================================================
# Tokenization for fallback keyword matching
# =============================================================================

_WORD_RE = re.compile(r"[a-z0-9]+")

def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


# =============================================================================
# Lane sanitization (from chat_loop.py)
# =============================================================================

_NULLABLE_STRING_KEYS = {
    "when",
    "parse_hint",
    "safety_notes",
    "expected_observation",
}

def _sanitize_lane_dict(item: Dict[str, Any]) -> Dict[str, Any]:
    """Make lane JSON robust against compiler emitting null for string-ish fields."""
    if not isinstance(item, dict):
        return item

    def fix(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if k in _NULLABLE_STRING_KEYS and v is None:
                    obj[k] = ""
                    continue
                fix(v)
        elif isinstance(obj, list):
            for x in obj:
                fix(x)

    fix(item)

    try:
        steps = item.get("steps")
        if isinstance(steps, list):
            for st in steps:
                if not isinstance(st, dict):
                    continue
                eas = st.get("evidence_actions")
                if isinstance(eas, list):
                    for ea in eas:
                        if not isinstance(ea, dict):
                            continue
                        req = ea.get("request")
                        if isinstance(req, dict) and req.get("parse_hint") is None:
                            req["parse_hint"] = ""
    except Exception:
        pass

    try:
        if (not item.get("entry_step_id")) and isinstance(item.get("steps"), list) and item["steps"]:
            first = item["steps"][0]
            if isinstance(first, dict) and first.get("step_id"):
                item["entry_step_id"] = first["step_id"]
    except Exception:
        pass

    return item


# =============================================================================
# Embedding computation
# =============================================================================

def compute_query_embedding_openai(
    query: str,
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
) -> List[float]:
    """Compute query embedding using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")
    
    if not api_key:
        raise ValueError("OpenAI API key required for embeddings")
    
    client = OpenAI(api_key=api_key)
    
    response = client.embeddings.create(
        model=model,
        input=query,
    )
    
    return response.data[0].embedding


def compute_query_embedding_local(
    query: str,
    model: str = "all-mpnet-base-v2",
) -> List[float]:
    """Compute query embedding using local sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers required. Install with: pip install sentence-transformers"
        )
    
    # Cache model to avoid reloading
    if not hasattr(compute_query_embedding_local, '_model_cache'):
        compute_query_embedding_local._model_cache = {}
    
    if model not in compute_query_embedding_local._model_cache:
        compute_query_embedding_local._model_cache[model] = SentenceTransformer(model)
    
    model_obj = compute_query_embedding_local._model_cache[model]
    embedding = model_obj.encode(query, convert_to_numpy=True)
    
    return embedding.tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    
    a_arr = np.array(a)
    b_arr = np.array(b)
    
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def compute_max_similarity(
    query_embedding: List[float],
    doc_embeddings: List[List[float]],
) -> float:
    """Compute maximum cosine similarity between query and any doc embedding."""
    if not query_embedding or not doc_embeddings:
        return 0.0
    
    max_sim = 0.0
    for doc_emb in doc_embeddings:
        sim = cosine_similarity(query_embedding, doc_emb)
        max_sim = max(max_sim, sim)
    
    return max_sim


# =============================================================================
# Embedding Lane Store
# =============================================================================

class EmbeddingLaneStore:
    """
    Lane store with semantic embedding-based retrieval.
    
    Uses pre-computed embeddings from procedure compilation (fact_embeddings field).
    Falls back to keyword matching if embeddings unavailable.
    """
    
    def __init__(
        self,
        lanes: List[ProcedureLane],
        embedding_model: str,
        use_openai: bool = False,
        api_key: Optional[str] = None,
    ):
        self.lanes = lanes
        self.embedding_model = embedding_model
        self.use_openai = use_openai
        self.api_key = api_key
        
        # Build keyword index for fallback
        self._keyword_index = self._build_keyword_index(lanes)
        
        # Count lanes with embeddings
        self.lanes_with_embeddings = sum(
            1 for lane in lanes 
            if hasattr(lane, 'fact_embeddings') and lane.fact_embeddings
        )
        
        print(f"   Lanes with embeddings: {self.lanes_with_embeddings}/{len(lanes)}")
        
        if self.lanes_with_embeddings == 0:
            print("   ⚠️  No lanes have embeddings - falling back to keyword matching")
            print("   Compile procedures with hybrid compiler to get embeddings")
    
    @staticmethod
    def load(
        lanes_path: str,
        embedding_model: str = "text-embedding-3-small",
        use_openai: bool = False,
        api_key: Optional[str] = None,
    ) -> "EmbeddingLaneStore":
        """Load lanes from JSONL/JSON file."""
        p = Path(lanes_path)
        if not p.exists():
            raise FileNotFoundError(f"Lane file not found: {p}")
        
        # Load raw JSON
        lanes_raw: List[Dict[str, Any]]
        if p.suffix.lower() in (".jsonl", ".jl"):
            lanes_raw = []
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lanes_raw.append(json.loads(line))
        else:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                lanes_raw = obj
            elif isinstance(obj, dict) and "lanes" in obj:
                lanes_raw = obj["lanes"]
            else:
                raise ValueError("Unsupported lane file format")
        
        # Parse lanes
        lanes: List[ProcedureLane] = []
        errors: List[str] = []
        
        for i, item in enumerate(lanes_raw):
            if not isinstance(item, dict):
                errors.append(f"Row {i}: not a JSON object")
                continue
            
            item = _sanitize_lane_dict(item)
            
            try:
                lane = ProcedureLane.model_validate(item)
                lanes.append(lane)
            except Exception as e:
                pid = item.get("procedure_id") or item.get("lane_id") or f"row:{i}"
                errors.append(f"Failed to parse lane {pid}: {e}")
        
        if not lanes:
            raise ValueError(f"No lanes could be loaded. Errors: {errors[:5]}")
        
        if errors and len(errors) <= 10:
            print(f"\n⚠️  {len(errors)} lanes failed to load:")
            for err in errors:
                print(f"   {err}")
        
        return EmbeddingLaneStore(
            lanes=lanes,
            embedding_model=embedding_model,
            use_openai=use_openai,
            api_key=api_key,
        )
    
    @staticmethod
    def _build_keyword_index(lanes: List[ProcedureLane]) -> List[Tuple[str, List[str]]]:
        """Build keyword index for fallback retrieval."""
        idx: List[Tuple[str, List[str]]] = []
        for lane in lanes:
            blob = " ".join([
                lane.lane_id,
                lane.doc_id,
                lane.doc_title,
                lane.title,
                lane.summary,
                " ".join([fs.fact for fs in (lane.fact_specs or [])]),
                " ".join([s.title or "" for s in (lane.steps[:3] if lane.steps else [])]),
                " ".join([s.instruction[:160] for s in (lane.steps[:2] if lane.steps else [])]),
                " ".join([t for t in (lane.tags or [])[:20]]),
                " ".join([sig for sig in (lane.signatures or [])[:20]]),
            ])
            idx.append((lane.lane_id, _tokenize(blob)))
        return idx
    
    def get_lane(self, lane_id: str) -> Optional[ProcedureLane]:
        """Get lane by ID."""
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane
        return None
    
    def _retrieve_with_keywords(
        self,
        query_text: str,
        cfg: RetrievalConfig,
        controller_state: Optional[ControllerState] = None,
    ) -> Tuple[List[ProcedureLane], Dict[str, float]]:
        """Fallback keyword-based retrieval."""
        q = _tokenize(query_text)
        scored: List[Tuple[float, ProcedureLane]] = []
        priors: Dict[str, float] = {}
        
        current_lane_id = controller_state.current_lane_id if controller_state else None
        
        # Get lanes with existing belief
        lanes_with_belief: Dict[str, float] = {}
        if controller_state and controller_state.belief.items:
            for item in controller_state.belief.items:
                if item.belief > 0.01:
                    lanes_with_belief[item.key] = item.belief
        
        lanes_to_keep: Dict[str, ProcedureLane] = {}
        
        for (lane_id, tokens) in self._keyword_index:
            lane = self.get_lane(lane_id)
            if lane is None:
                continue
            
            score = _jaccard(q, tokens)
            
            # Boost lanes with existing belief
            if lane_id in lanes_with_belief:
                boost_factor = 1.5 + (lanes_with_belief[lane_id] * 2.0)
                score *= boost_factor
                lanes_to_keep[lane_id] = lane
            
            # Extra boost for current lane
            if cfg.prefer_current_lane and current_lane_id == lane_id:
                score *= 2.0
                lanes_to_keep[lane_id] = lane
            
            priors[lane_id] = score
            
            if score >= cfg.min_score:
                scored.append((score, lane))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Build result
        result_lanes = [l for _, l in scored[:cfg.top_k]]
        
        # Force inclusion of lanes with belief
        for lane_id, lane in lanes_to_keep.items():
            if lane not in result_lanes:
                result_lanes.insert(0, lane)
        
        result_lanes = result_lanes[:cfg.top_k]
        
        if not result_lanes:
            result_lanes = self.lanes[:max(1, cfg.top_k)]
        
        return result_lanes, priors
    
    def retrieve(
        self,
        query_text: str,
        cfg: RetrievalConfig,
        *,
        controller_state: Optional[ControllerState] = None,
    ) -> Tuple[List[ProcedureLane], Dict[str, float]]:
        """
        Retrieve lanes using semantic embeddings.
        
        Falls back to keyword matching if:
        - No lanes have embeddings
        - Query embedding computation fails
        
        Returns:
            (lanes, priors_dict) where priors are similarity scores
        """
        # Check if we have any embeddings
        if self.lanes_with_embeddings == 0:
            return self._retrieve_with_keywords(query_text, cfg, controller_state)
        
        # Compute query embedding
        try:
            if self.use_openai:
                query_embedding = compute_query_embedding_openai(
                    query_text,
                    model=self.embedding_model,
                    api_key=self.api_key,
                )
            else:
                query_embedding = compute_query_embedding_local(
                    query_text,
                    model=self.embedding_model,
                )
        except Exception as e:
            print(f"⚠️  Embedding computation failed: {e}")
            print("   Falling back to keyword matching")
            return self._retrieve_with_keywords(query_text, cfg, controller_state)
        
        # Score lanes by semantic similarity
        scored: List[Tuple[float, ProcedureLane]] = []
        priors: Dict[str, float] = {}
        
        current_lane_id = controller_state.current_lane_id if controller_state else None
        
        # Get lanes with existing belief
        lanes_with_belief: Dict[str, float] = {}
        if controller_state and controller_state.belief.items:
            for item in controller_state.belief.items:
                if item.belief > 0.01:
                    lanes_with_belief[item.key] = item.belief
        
        lanes_to_keep: Dict[str, ProcedureLane] = {}
        
        for lane in self.lanes:
            # Use embeddings if available, otherwise fall back to keywords
            if hasattr(lane, 'fact_embeddings') and lane.fact_embeddings:
                score = compute_max_similarity(query_embedding, lane.fact_embeddings)
            else:
                # Fallback to keyword matching for this lane
                q_tokens = _tokenize(query_text)
                lane_tokens = dict(self._keyword_index).get(lane.lane_id, [])
                score = _jaccard(q_tokens, lane_tokens) * 0.5  # Scale down keyword scores
            
            # Boost lanes with existing belief
            if lane.lane_id in lanes_with_belief:
                boost_factor = 1.5 + (lanes_with_belief[lane.lane_id] * 2.0)
                score *= boost_factor
                lanes_to_keep[lane.lane_id] = lane
            
            # Extra boost for current lane
            if cfg.prefer_current_lane and current_lane_id == lane.lane_id:
                score *= 2.0
                lanes_to_keep[lane.lane_id] = lane
            
            priors[lane.lane_id] = score
            
            if score >= cfg.min_score:
                scored.append((score, lane))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Build result
        result_lanes = [l for _, l in scored[:cfg.top_k]]
        
        # Force inclusion of lanes with belief
        for lane_id, lane in lanes_to_keep.items():
            if lane not in result_lanes:
                result_lanes.insert(0, lane)
        
        result_lanes = result_lanes[:cfg.top_k]
        
        if not result_lanes:
            result_lanes = self.lanes[:max(1, cfg.top_k)]
        
        return result_lanes, priors