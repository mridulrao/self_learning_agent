#!/usr/bin/env python3
"""
hybrid_fact_matching.py

Core hybrid retrieval system for fact-to-lane matching.
Uses BM25 (lexical) + Semantic Embeddings + Cross-Encoder reranking.

This module provides:
1. Compile-time indexing: compute BM25 keywords + dense embeddings
2. Runtime matching: score extracted facts against lane indices
3. Fusion: combine multiple signals (RRF or weighted)
4. Optional reranking: cross-encoder for high-accuracy disambiguation

Design principle:
- Semantic matching handles synonyms/paraphrasing
- BM25 catches exact technical terms (error codes, product names)
- Cross-encoder provides final high-precision reranking when needed
"""

from __future__ import annotations

import logging
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Tokenization for BM25
# =============================================================================

# Common English stopwords
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'should', 'could', 'may', 'might', 'must', 'can',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
    'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
    'why', 'how', 'and', 'or', 'but', 'if', 'then', 'for', 'of',
    'to', 'from', 'in', 'on', 'at', 'by', 'with', 'about', 'as',
}


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenize text for BM25 indexing.
    
    Args:
        text: Input text
        remove_stopwords: Whether to filter out stopwords
        
    Returns:
        List of tokens (lowercase, no punctuation, length > 2)
    """
    if not text:
        return []
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Split on whitespace
    tokens = text.split()
    
    # Filter
    if remove_stopwords:
        tokens = [t for t in tokens if t and t not in STOPWORDS and len(t) > 2]
    else:
        tokens = [t for t in tokens if t and len(t) > 2]
    
    return tokens


# =============================================================================
# BM25 Scoring
# =============================================================================

def compute_bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    k1: float = 1.5,
    b: float = 0.75,
    avg_doc_len: float = 50.0,
) -> float:
    """
    Compute BM25 score for a single document.
    
    Args:
        query_tokens: Tokenized query (extracted facts)
        doc_tokens: Tokenized document (lane keywords)
        k1: BM25 parameter (term saturation)
        b: BM25 parameter (length normalization)
        avg_doc_len: Average document length in corpus
        
    Returns:
        BM25 score (higher is better)
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    
    doc_len = len(doc_tokens)
    doc_term_freq = Counter(doc_tokens)
    
    score = 0.0
    
    for term in query_tokens:
        if term not in doc_term_freq:
            continue
        
        tf = doc_term_freq[term]
        
        # Simplified IDF (assume df=1 for matched terms, corpus_size=100)
        # In production, you'd compute real IDF from full corpus
        idf = np.log((100 - 1 + 0.5) / (1 + 0.5))
        
        # BM25 formula
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
        
        score += idf * (numerator / denominator)
    
    return score


def normalize_bm25_score(score: float, max_theoretical: float = 10.0) -> float:
    """
    Normalize BM25 score to [0, 1] range.
    
    Args:
        score: Raw BM25 score
        max_theoretical: Rough upper bound for normalization
        
    Returns:
        Normalized score in [0, 1]
    """
    return min(1.0, score / max_theoretical)


# =============================================================================
# Semantic Similarity (Dense Retrieval)
# =============================================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First embedding vector
        b: Second embedding vector
        
    Returns:
        Cosine similarity in [-1, 1] (typically [0, 1] for embeddings)
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    
    a_arr = np.array(a)
    b_arr = np.array(b)
    
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def compute_max_semantic_similarity(
    query_embedding: List[float],
    doc_embeddings: List[List[float]],
) -> float:
    """
    Compute maximum cosine similarity between query and any document embedding.
    
    Args:
        query_embedding: Query embedding vector
        doc_embeddings: List of document embedding vectors
        
    Returns:
        Maximum similarity score in [0, 1]
    """
    if not query_embedding or not doc_embeddings:
        return 0.0
    
    max_sim = 0.0
    
    for doc_emb in doc_embeddings:
        sim = cosine_similarity(query_embedding, doc_emb)
        max_sim = max(max_sim, sim)
    
    return max_sim


# =============================================================================
# Reciprocal Rank Fusion (RRF)
# =============================================================================

def reciprocal_rank_fusion(
    scores_dict: Dict[str, Dict[str, float]],
    k: int = 60,
) -> Dict[str, float]:
    """
    Combine multiple ranking sources using Reciprocal Rank Fusion.
    
    RRF formula: score(d) = sum over all rankers r of: 1 / (k + rank_r(d))
    
    Args:
        scores_dict: {
            "source_name": {item_id: score, ...},
            ...
        }
        k: RRF constant (default 60, standard in literature)
        
    Returns:
        {item_id: fused_score} - higher is better
    """
    if not scores_dict:
        return {}
    
    # Collect all unique item IDs
    all_items = set()
    for source_scores in scores_dict.values():
        all_items.update(source_scores.keys())
    
    fused_scores = {item_id: 0.0 for item_id in all_items}
    
    # For each ranking source
    for source_name, source_scores in scores_dict.items():
        if not source_scores:
            continue
        
        # Sort by score descending to get ranks
        ranked = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Apply RRF formula
        for rank, (item_id, score) in enumerate(ranked, start=1):
            fused_scores[item_id] += 1.0 / (k + rank)
    
    return fused_scores


def weighted_score_fusion(
    scores_dict: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Combine multiple scoring sources using weighted average.
    
    Args:
        scores_dict: {source_name: {item_id: score}}
        weights: {source_name: weight} - defaults to equal weights
        
    Returns:
        {item_id: weighted_score}
    """
    if not scores_dict:
        return {}
    
    # Default to equal weights
    if weights is None:
        weights = {name: 1.0 / len(scores_dict) for name in scores_dict.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        return {}
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Collect all items
    all_items = set()
    for source_scores in scores_dict.values():
        all_items.update(source_scores.keys())
    
    fused_scores = {}
    
    for item_id in all_items:
        weighted_sum = 0.0
        
        for source_name, source_scores in scores_dict.items():
            score = source_scores.get(item_id, 0.0)
            weight = weights.get(source_name, 0.0)
            weighted_sum += score * weight
        
        fused_scores[item_id] = weighted_sum
    
    return fused_scores


# =============================================================================
# Cross-Encoder Reranking
# =============================================================================

_cross_encoder_model = None


def get_cross_encoder():
    """
    Lazy load cross-encoder model (singleton pattern).
    
    Returns:
        CrossEncoder model or None if unavailable
    """
    global _cross_encoder_model
    
    if _cross_encoder_model is None:
        try:
            from sentence_transformers import CrossEncoder
            
            # ms-marco-MiniLM-L-6-v2: fast, accurate for passage ranking
            _cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Loaded cross-encoder model: ms-marco-MiniLM-L-6-v2")
            
        except ImportError:
            logger.warning("sentence-transformers not installed. Cross-encoder unavailable.")
            _cross_encoder_model = "unavailable"
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}")
            _cross_encoder_model = "unavailable"
    
    return _cross_encoder_model if _cross_encoder_model != "unavailable" else None


def cross_encoder_rerank(
    query: str,
    candidates: List[Tuple[str, str]],
    top_k: Optional[int] = None,
) -> Dict[str, float]:
    """
    Rerank candidates using cross-encoder.
    
    Args:
        query: Query text (e.g., extracted facts as sentence)
        candidates: [(item_id, document_text), ...]
        top_k: Return only top-k results (None = all)
        
    Returns:
        {item_id: reranked_score} - higher is better
    """
    model = get_cross_encoder()
    
    if model is None or not candidates:
        return {}
    
    try:
        # Prepare (query, document) pairs
        pairs = [(query, doc) for _, doc in candidates]
        
        # Score all pairs
        scores = model.predict(pairs)
        
        # Build result dict
        results = {
            item_id: float(score)
            for (item_id, _), score in zip(candidates, scores)
        }
        
        # Optionally limit to top-k
        if top_k is not None:
            sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
            results = dict(sorted_items[:top_k])
        
        return results
        
    except Exception as e:
        logger.warning(f"Cross-encoder reranking failed: {e}")
        return {}


# =============================================================================
# Compile-Time: Index Generation
# =============================================================================

def extract_keywords_from_texts(
    texts: List[str],
    top_k: int = 100,
) -> List[str]:
    """
    Extract top-k keywords from a list of texts for BM25 indexing.
    
    Args:
        texts: List of text strings
        top_k: Number of top keywords to return (by frequency)
        
    Returns:
        List of keywords (deduplicated, ranked by frequency)
    """
    all_tokens = []
    
    for text in texts:
        all_tokens.extend(tokenize(text, remove_stopwords=True))
    
    # Count frequencies
    token_counts = Counter(all_tokens)
    
    # Return top-k by frequency
    return [token for token, _ in token_counts.most_common(top_k)]


def compute_embeddings_batch(
    texts: List[str],
    embedding_model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """
    Compute embeddings for a batch of texts.
    
    Args:
        texts: List of text strings
        embedding_model: OpenAI embedding model name
        api_key: OpenAI API key (required)
        
    Returns:
        List of embedding vectors (same order as input)
    """
    if not texts or not api_key:
        return []
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        response = client.embeddings.create(
            model=embedding_model,
            input=texts
        )
        
        return [emb.embedding for emb in response.data]
        
    except Exception as e:
        logger.error(f"Failed to compute embeddings: {e}")
        return []


def build_lane_retrieval_index(
    fact_texts: List[str],
    embedding_model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    max_keywords: int = 100,
    max_embeddings: int = 40,
) -> Dict[str, Any]:
    """
    Build complete retrieval index for a lane.
    
    Args:
        fact_texts: List of fact-related texts (title, summary, steps, etc.)
        embedding_model: OpenAI embedding model
        api_key: OpenAI API key (for embeddings)
        max_keywords: Max BM25 keywords to extract
        max_embeddings: Max texts to embed (for cost control)
        
    Returns:
        {
            "fact_keywords": [str],           # BM25 index
            "fact_embeddings": [              # Dense embeddings
                {"text": str, "embedding": [float], "index": int}
            ],
            "fact_descriptions": [str]        # For cross-encoder
        }
    """
    if not fact_texts:
        return {
            "fact_keywords": [],
            "fact_embeddings": [],
            "fact_descriptions": [],
        }
    
    # Dedupe texts
    fact_texts = list(dict.fromkeys(fact_texts))
    
    # === BM25 Keywords ===
    keywords = extract_keywords_from_texts(fact_texts, top_k=max_keywords)
    
    # === Dense Embeddings ===
    embeddings_input = fact_texts[:max_embeddings]
    embeddings = compute_embeddings_batch(
        embeddings_input,
        embedding_model=embedding_model,
        api_key=api_key
    )
    
    fact_embeddings = []
    if embeddings:
        # Store only the embedding vectors (not metadata)
        fact_embeddings = [emb for emb in embeddings if emb]
    
    # === Cross-Encoder Descriptions ===
    # Keep top texts for reranking
    descriptions = fact_texts[:30]
    
    return {
        "fact_keywords": keywords,
        "fact_embeddings": fact_embeddings,
        "fact_descriptions": descriptions,
    }


# =============================================================================
# Runtime: Hybrid Fact Matching
# =============================================================================

def format_facts_as_query(facts: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Convert extracted facts to query text and tokens.
    
    Args:
        facts: {fact_key: ExtractedFact} or {fact_key: value}
        
    Returns:
        (query_text, query_tokens)
    """
    fact_texts = []
    
    for key, value in facts.items():
        # Handle ExtractedFact objects
        if hasattr(value, 'value'):
            val = value.value
        else:
            val = value
        
        if val is None:
            continue
        
        # Format as natural language
        key_readable = key.replace('_', ' ')
        
        if isinstance(val, bool):
            status = "yes" if val else "no"
            fact_texts.append(f"{key_readable} {status}")
        elif isinstance(val, str):
            fact_texts.append(f"{key_readable} is {val}")
        else:
            fact_texts.append(f"{key_readable} {val}")
    
    query_text = ". ".join(fact_texts)
    query_tokens = tokenize(query_text, remove_stopwords=True)
    
    return query_text, query_tokens


def compute_hybrid_match_score(
    facts: Dict[str, Any],
    lane_index: Dict[str, Any],
    query_embedding: Optional[List[float]] = None,
    fusion_method: str = "weighted",  # "weighted" or "rrf"
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute hybrid match score: BM25 + Semantic.
    
    Args:
        facts: Extracted facts {fact_key: value}
        lane_index: Lane retrieval index from build_lane_retrieval_index()
        query_embedding: Pre-computed query embedding (optional)
        fusion_method: "weighted" or "rrf"
        weights: Custom weights for fusion (default: {"bm25": 0.3, "semantic": 0.7})
        
    Returns:
        (fused_score, component_scores_dict)
    """
    if not facts:
        return 0.0, {}
    
    query_text, query_tokens = format_facts_as_query(facts)
    
    component_scores = {}
    
    # === BM25 Score ===
    lane_keywords = lane_index.get("fact_keywords", [])
    
    if lane_keywords:
        bm25_raw = compute_bm25_score(query_tokens, lane_keywords)
        bm25_norm = normalize_bm25_score(bm25_raw)
        component_scores["bm25"] = bm25_norm
    
    # === Semantic Score ===
    lane_embeddings_list = lane_index.get("fact_embeddings", [])
    
    if query_embedding and lane_embeddings_list:
        # Embeddings are now stored directly as vectors (not dicts)
        semantic_score = compute_max_semantic_similarity(query_embedding, lane_embeddings_list)
        component_scores["semantic"] = semantic_score
    
    # === Fusion ===
    if not component_scores:
        return 0.0, {}
    
    if fusion_method == "rrf":
        # For RRF, we need multiple items, but we only have one lane
        # So we use weighted average as fallback
        fusion_method = "weighted"
    
    if fusion_method == "weighted":
        if weights is None:
            # Default weights: favor semantic (better for concept matching)
            weights = {"bm25": 0.3, "semantic": 0.7}
        
        fused = 0.0
        total_weight = 0.0
        
        for component, score in component_scores.items():
            weight = weights.get(component, 0.0)
            fused += score * weight
            total_weight += weight
        
        if total_weight > 0:
            fused /= total_weight
    else:
        # Fallback: simple average
        fused = sum(component_scores.values()) / len(component_scores)
    
    return fused, component_scores


def hybrid_rerank_with_cross_encoder(
    query_text: str,
    candidates: Dict[str, Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Rerank top candidates using cross-encoder.
    
    Args:
        query_text: Query text (formatted facts)
        candidates: {lane_id: lane_index} - must have "fact_descriptions"
        top_k: Number of results to return
        
    Returns:
        {lane_id: reranked_score}
    """
    if not candidates:
        return {}
    
    # Build (lane_id, description) pairs
    pairs = []
    
    for lane_id, lane_index in candidates.items():
        descriptions = lane_index.get("fact_descriptions", [])
        if descriptions:
            # Concatenate top descriptions
            desc_text = " ".join(descriptions[:5])
            pairs.append((lane_id, desc_text))
    
    if not pairs:
        return {}
    
    # Rerank
    return cross_encoder_rerank(query_text, pairs, top_k=top_k)


# =============================================================================
# Scoring Helper: Convert Match Score to Belief Delta
# =============================================================================

def match_score_to_belief_delta(
    match_score: float,
    phase: str = "triage",
) -> float:
    """
    Convert hybrid match score [0, 1] to belief delta.
    
    UPDATED: More aggressive scaling to ensure facts contribute to belief
    
    Args:
        match_score: Hybrid retrieval score in [0, 1]
        phase: "triage" or "execute" (affects capping)
        
    Returns:
        Belief delta (capped based on phase)
    """
    # More aggressive tiered scaling
    if match_score >= 0.80:
        delta = 0.45  # Strong match
    elif match_score >= 0.65:
        delta = 0.35  # Good match
    elif match_score >= 0.50:
        delta = 0.25  # Moderate match
    elif match_score >= 0.35:
        delta = 0.18  # Weak match
    elif match_score >= 0.20:  # LOWERED FROM 0.25
        delta = 0.10  # Very weak match (still counts!)
    elif match_score >= 0.10:  # NEW TIER
        delta = 0.05  # Minimal match
    else:
        delta = 0.0
    
    # Phase-based capping
    if phase.lower() == "triage":
        # Looser cap during exploration
        delta = max(-0.35, min(0.45, delta))
    else:
        # Tighter cap during execution
        delta = max(-0.20, min(0.35, delta))
    
    return delta


# =============================================================================
# Debugging / Logging Utilities
# =============================================================================

def format_match_reason(
    component_scores: Dict[str, float],
    fused_score: float,
    reranked_score: Optional[float] = None,
) -> str:
    """
    Format match reasoning for debug logging.
    
    Args:
        component_scores: {"bm25": 0.6, "semantic": 0.8}
        fused_score: Fused score from components
        reranked_score: Cross-encoder score (if used)
        
    Returns:
        Human-readable reason string
    """
    parts = []
    
    for component, score in component_scores.items():
        parts.append(f"{component}={score:.3f}")
    
    parts.append(f"fused={fused_score:.3f}")
    
    if reranked_score is not None:
        parts.append(f"cross_enc={reranked_score:.3f}")
    
    return "; ".join(parts)