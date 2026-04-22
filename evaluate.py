"""
evaluate.py
===========
AIMS KTT Hackathon 2026 | G5 | S2.T1.3
Made in Rwanda Content Recommender

Evaluation Metrics:
  1. NDCG@5 — ranking quality score
  2. Local Presence Rate — % queries with local product in top 3

Author : Sltanu Kifile Alemu
Date   : 22 April 2026
"""

import pandas as pd
import numpy as np
from recommender import load_catalog, build_index, recommend

# ── Load Data ─────────────────────────────────────────────────
print("=" * 55)
print("  EVALUATION — G5 S2.T1.3")
print("=" * 55)

df         = load_catalog("catalog.csv")
queries_df = pd.read_csv("queries.csv")
vectorizer, tfidf_matrix = build_index(df)

print(f"  Catalog loaded : {len(df)} products")
print(f"  Queries loaded : {len(queries_df)} queries")
print()

# ── NDCG@5 ───────────────────────────────────────────────────
def dcg_at_k(scores, k=5):
    """
    Discounted Cumulative Gain at k.
    Higher ranked relevant items score more.
    """
    scores = np.array(scores)[:k]
    gains  = scores / np.log2(np.arange(2, len(scores) + 2))
    return gains.sum()

def ndcg_at_k(results, k=5):
    """
    Normalized Discounted Cumulative Gain at k.
    Compares actual ranking to ideal ranking.
    Score of 1.0 = perfect ranking.
    """
    scores  = results["score"].tolist()[:k]
    ideal   = sorted(scores, reverse=True)
    actual  = dcg_at_k(scores, k)
    perfect = dcg_at_k(ideal, k)
    if perfect == 0:
        return 0.0
    return actual / perfect

# ── Local Presence Rate ───────────────────────────────────────
def local_presence(results, top_n=3):
    """
    Check if at least one local product in top N.
    Returns 1 if yes, 0 if no.
    """
    top = results.head(top_n)
    return int(top["origin_district"].notna().any())

# ── Run Evaluation ────────────────────────────────────────────
print("  Running evaluation on all 120 queries...")
print()

ndcg_scores           = []
local_presence_scores = []

for _, row in queries_df.iterrows():
    query = row["query_text"]
    try:
        results = recommend(query, df, vectorizer, tfidf_matrix)
        ndcg_scores.append(ndcg_at_k(results, k=5))
        local_presence_scores.append(local_presence(results, top_n=3))
    except:
        ndcg_scores.append(0.0)
        local_presence_scores.append(0)

# ── Overall Results ───────────────────────────────────────────
avg_ndcg        = np.mean(ndcg_scores)
local_pres_rate = np.mean(local_presence_scores) * 100

print("=" * 55)
print("  EVALUATION RESULTS")
print("=" * 55)
print(f"  NDCG@5              : {avg_ndcg:.4f}")
print(f"  Local Presence Rate : {local_pres_rate:.1f}%")
print()

# ── Breakdown by Language ─────────────────────────────────────
print("  Breakdown by language:")
for lang in ["en", "fr", "mixed"]:
    lang_queries = queries_df[queries_df["language"] == lang]
    lang_ndcg = []
    lang_lp   = []
    for _, row in lang_queries.iterrows():
        query = row["query_text"]
        try:
            results = recommend(query, df, vectorizer, tfidf_matrix)
            lang_ndcg.append(ndcg_at_k(results, k=5))
            lang_lp.append(local_presence(results, top_n=3))
        except:
            lang_ndcg.append(0.0)
            lang_lp.append(0)
    print(f"    {lang.upper():<6} → NDCG@5: {np.mean(lang_ndcg):.4f} | Local Presence: {np.mean(lang_lp)*100:.1f}%")

print()
print("=" * 55)
print("  FINAL SCORES FOR YOUR VIDEO INTRO")
print("=" * 55)
print(f"  NDCG@5              = {avg_ndcg:.4f}")
print(f"  Local Presence Rate = {local_pres_rate:.1f}%")
print("=" * 55)
