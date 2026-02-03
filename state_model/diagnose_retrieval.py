#!/usr/bin/env python3
"""
diagnose_retrieval.py

Diagnose why retrieval is returning 0.0 scores.
"""

import json
import sys
from pathlib import Path

def check_embeddings(jsonl_path: str, query: str = "setup mobile device company email"):
    """Check if lanes have valid embeddings."""
    
    print(f"Checking: {jsonl_path}")
    print(f"Query: {query}\n")
    
    with open(jsonl_path) as f:
        lanes = [json.loads(line) for line in f if line.strip()]
    
    print(f"Total lanes: {len(lanes)}\n")
    
    # Check a few lanes
    for i, lane in enumerate(lanes[:5]):
        title = lane.get('title', 'NO TITLE')
        lane_id = lane.get('procedure_id', 'NO ID')
        
        print(f"\n{'='*70}")
        print(f"Lane {i+1}: {title}")
        print(f"ID: {lane_id}")
        print(f"{'='*70}")
        
        # Check hybrid fields
        has_keywords = 'fact_keywords' in lane
        has_embeddings = 'fact_embeddings' in lane
        has_descriptions = 'fact_descriptions' in lane
        
        print(f"Has fact_keywords: {has_keywords}")
        print(f"Has fact_embeddings: {has_embeddings}")
        print(f"Has fact_descriptions: {has_descriptions}")
        
        if has_keywords:
            keywords = lane['fact_keywords']
            print(f"  Keywords count: {len(keywords)}")
            if keywords:
                print(f"  Sample keywords: {keywords[:10]}")
        
        if has_embeddings:
            embeddings = lane['fact_embeddings']
            print(f"  Embeddings count: {len(embeddings)}")
            if embeddings:
                first_emb = embeddings[0]
                if isinstance(first_emb, list):
                    print(f"  Embedding dimensions: {len(first_emb)}")
                    print(f"  Sample values: {first_emb[:5]}")
                else:
                    print(f"  ERROR: Embedding is {type(first_emb)}, not list!")
        
        if has_descriptions:
            descriptions = lane['fact_descriptions']
            print(f"  Descriptions count: {len(descriptions)}")
            if descriptions:
                print(f"  Sample: {descriptions[0][:100]}")
    
    # Check the "Setup Mobile Device" lane specifically
    print(f"\n{'='*70}")
    print("Searching for 'mobile device' lane...")
    print(f"{'='*70}")
    
    mobile_lanes = [
        lane for lane in lanes 
        if 'mobile' in lane.get('title', '').lower() 
        and 'email' in lane.get('title', '').lower()
    ]
    
    if mobile_lanes:
        lane = mobile_lanes[0]
        print(f"\nFound: {lane.get('title')}")
        print(f"ID: {lane.get('procedure_id')}")
        
        if lane.get('fact_embeddings'):
            print(f"Has {len(lane['fact_embeddings'])} embeddings")
            
            # Try to compute similarity manually
            try:
                import numpy as np
                from embedding_retrieval import (
                    compute_query_embedding_openai,
                    cosine_similarity,
                    compute_max_similarity
                )
                
                print("\nTesting similarity calculation...")
                
                # Compute query embedding
                api_key = input("Enter OpenAI API key (or press Enter to skip): ").strip()
                if api_key:
                    query_emb = compute_query_embedding_openai(
                        query, 
                        model="text-embedding-3-small",
                        api_key=api_key
                    )
                    
                    print(f"Query embedding dims: {len(query_emb)}")
                    
                    # Compute similarity
                    max_sim = compute_max_similarity(query_emb, lane['fact_embeddings'])
                    print(f"Max similarity: {max_sim:.4f}")
                    
                    if max_sim < 0.01:
                        print("⚠️  Similarity is very low!")
                        print("Possible issues:")
                        print("  - Query embedding model mismatch")
                        print("  - Lane embeddings are wrong format")
                        print("  - Embeddings are from different model")
            except Exception as e:
                print(f"Error testing similarity: {e}")
    else:
        print("⚠️  No mobile device lane found!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_retrieval.py <procedures.jsonl>")
        sys.exit(1)
    
    check_embeddings(sys.argv[1])