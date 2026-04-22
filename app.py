import gradio as gr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommender import load_catalog, build_index, recommend, translate_query

# ── Load and Build Index Once ─────────────────────────────────
df = load_catalog("catalog.csv")
vectorizer, tfidf_matrix = build_index(df)

# ── Recommend Function for Gradio ─────────────────────────────
def search(query):
    if not query.strip():
        return "Please enter a search query."

    results = recommend(query, df, vectorizer, tfidf_matrix)
    translated = translate_query(query)

    output = []
    output.append(f"Query      : {query}")
    output.append(f"Translated : {translated}")
    output.append(f"Query time : {results['query_time_ms'].iloc[0]} ms")
    output.append("=" * 55)

    for i, row in results.iterrows():
        boosted = " LOCAL BOOST" if row["boosted"] else ""
        output.append(f"\nRank {i+1}{boosted}")
        output.append(f"Product    : {row['title']}")
        output.append(f"Category   : {row['category']}")
        output.append(f"Material   : {row['material']}")
        output.append(f"District   : {row['origin_district']}")
        output.append(f"Price(RWF) : {row['price_rwf']:,}")
        output.append(f"Score      : {row['score']:.4f}")
        output.append("-" * 55)

    return "\n".join(output)

# ── Gradio Interface ──────────────────────────────────────────
demo = gr.Interface(
    fn          = search,
    inputs      = gr.Textbox(
        label       = "Search Query",
        placeholder = "Try: leather boots / cadeau en cuir pour femme / agaseke na impano",
        lines       = 1
    ),
    outputs     = gr.Textbox(
        label = "Top 5 Made-in-Rwanda Products",
        lines = 30
    ),
    title       = "Made in Rwanda Content Recommender",
    description = """
### AIMS KTT Hackathon 2026 | G5 | S2.T1.3

A niche-first recommender that surfaces local Made-in-Rwanda 
products over global brands.

**Supports:** English · French · Kinyarwanda

**Results:** NDCG@5 = 0.9833 | Local Presence Rate = 100%

**Try these queries:**
- English: `leather boots` · `handmade basket` · `beaded necklace`
- French: `cadeau en cuir pour femme` · `sac en cuir` · `panier tresse`
- Kinyarwanda: `agaseke na impano` · `uruhu na inkweto`
    """,
    examples    = [
        ["leather boots"],
        ["cadeau en cuir pour femme"],
        ["agaseke na impano"],
        ["handmade basket"],
        ["beaded necklace"],
        ["sac en cuir"],
    ],
    theme       = gr.themes.Soft()
)

demo.launch()
