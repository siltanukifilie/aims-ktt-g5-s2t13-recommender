"""
recommender.py
==============
AIMS KTT Hackathon 2026 | G5 | S2.T1.3
Made in Rwanda Content Recommender

Supports: English, French, Kinyarwanda, code-switched queries

Author : Sltanu Kifile Alemu
Date   : 22 April 2026
"""

import argparse
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Constants ─────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.05
TOP_N                = 5

# ── French → English ──────────────────────────────────────────
FR_TO_EN = {
    "cuir"          : "leather leather leather",
    "bottes"        : "boots leather",
    "sac"           : "bag handbag",
    "robe"          : "dress apparel",
    "collier"       : "necklace jewelry",
    "perles"        : "beads jewelry",
    "bracelet"      : "bracelet jewelry",
    "boucles"       : "earrings jewelry",
    "oreilles"      : "earrings",
    "panier"        : "basket basketry",
    "tresse"        : "woven basket",
    "tissee"        : "woven",
    "artisanal"     : "handmade local",
    "artisanat"     : "crafts handmade",
    "bijoux"        : "jewelry jewellery",
    "africain"      : "african",
    "africaine"     : "african",
    "rwandais"      : "rwandan rwanda",
    "rwandaise"     : "rwandan rwanda",
    "cadeau"        : "gift",
    "femme"         : "woman leather bag",
    "homme"         : "man",
    "fait"          : "handmade",
    "main"          : "handmade",
    "sandales"      : "sandals leather",
    "portefeuille"  : "wallet leather",
    "ceinture"      : "belt leather",
    "sculpture"     : "sculpture wood",
    "bois"          : "wood wooden",
    "argile"        : "clay pottery",
    "pot"           : "pot clay",
    "bol"           : "bowl wooden",
    "chemise"       : "shirt apparel",
    "coton"         : "cotton apparel",
    "tissu"         : "fabric kitenge",
    "imprimee"      : "print african",
    "traditionnel"  : "traditional",
    "traditionnelle": "traditional",
    "natte"         : "mat sisal",
    "bambou"        : "bamboo basket",
    "raffia"        : "raffia basket",
    "papyrus"       : "papyrus basket",
    "cuivre"        : "copper jewellery",
    "argent"        : "silver jewellery",
    "perle"         : "bead jewelry",
    "art"           : "art wall",
    "mural"         : "wall art",
    "poterie"       : "pottery clay",
    "en"            : "",
    "de"            : "",
    "pour"          : "",
    "le"            : "",
    "la"            : "",
    "les"           : "",
    "un"            : "",
    "une"           : "",
    "du"            : "",
    "des"           : "",
    "et"            : "",
    "a"             : "",
}

# ── Kinyarwanda → English ─────────────────────────────────────
RW_TO_EN = {
    # Leather & accessories
    "uruhu"         : "leather leather",
    "inkweto"       : "sandals leather",
    "umufuko"       : "bag handbag leather",
    "agaheto"       : "belt leather",
    "agasanduku"    : "wallet leather",
    # Clothing
    "impuzu"        : "dress clothing apparel",
    "ishati"        : "shirt apparel",
    "ikoti"         : "jacket apparel",
    "umwitero"      : "sweater apparel",
    # Basketry
    "agaseke"       : "basket basketry peace",
    "ingobyi"       : "basket woven",
    "indobo"        : "basket storage",
    "isahane"       : "basket sisal",
    # Jewellery
    "urunigi"       : "necklace jewelry beads",
    "icumu"         : "bracelet jewelry",
    "amasato"       : "earrings jewelry",
    "impande"       : "jewelry beads",
    # Home decor
    "inkono"        : "pot clay pottery",
    "icyombo"       : "bowl wooden",
    "igitanda"      : "wood wooden",
    "urugi"         : "door wood",
    "imigongo"      : "wall art imigongo traditional",
    # General
    "ibikoranabuhanga" : "crafts handmade",
    "ubugeni"       : "crafts handmade local",
    "umuganura"     : "gift traditional",
    "impano"        : "gift handmade",
    "abagore"       : "women cooperative",
    "umubare"       : "price",
    "gura"          : "buy",
    "ibicuruzwa"    : "products local",
    "ubworozi"      : "leather cowhide",
    "inka"          : "cow cowhide leather",
    "imyenda"       : "clothing apparel",
    "ifeza"         : "silver jewelry",
    "umuringa"      : "copper jewelry",
    "igiti"         : "wood wooden",
    "urubabi"       : "sisal basket",
    "ubwoya"        : "wool apparel",
    "na"            : "",
    "ni"            : "",
    "mu"            : "",
    "ku"            : "",
    "ya"            : "",
    "wa"            : "",
    "za"            : "",
    "ba"            : "",
    "cy"            : "",
    "ry"            : "",
}

# ── Translate Query ───────────────────────────────────────────
def translate_query(q):
    """
    Translate French/Kinyarwanda/mixed query to English.
    Checks both FR and RW dictionaries word by word.
    Unknown words kept as-is to handle English + mixed.
    """
    words = q.lower().split()
    translated = []
    for word in words:
        if word in FR_TO_EN:
            english = FR_TO_EN[word]
        elif word in RW_TO_EN:
            english = RW_TO_EN[word]
        else:
            english = word
        if english:
            translated.append(english)
    result = " ".join(translated)
    return result if result.strip() else q

# ── Load Catalog ──────────────────────────────────────────────
def load_catalog(path="catalog.csv"):
    """Load the Made-in-Rwanda product catalog."""
    df = pd.read_csv(path)
    df["is_local"] = df["origin_district"].notna()
    df["text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["category"].fillna("") + " " +
        df["material"].fillna("") + " " +
        df["origin_district"].fillna("")
    )
    return df

# ── Build TF-IDF Index ────────────────────────────────────────
def build_index(df):
    """
    Build TF-IDF matrix on combined product text.

    Why TF-IDF over sentence embeddings:
      - CPU-only, no GPU needed
      - Query time under 250ms on free Colab CPU
      - No paid API required
      - Fast, interpretable, debuggable

    Weakness:
      - Misses purely semantic queries like gift for mom
        where no keyword matches product text directly
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words=None
    )
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    return vectorizer, tfidf_matrix

# ── Core Recommend Function ───────────────────────────────────
def recommend(q, df, vectorizer, tfidf_matrix, top_n=TOP_N):
    """
    Core recommendation function.

    Steps:
      1. Translate FR/RW/mixed query to English
      2. Vectorize query using TF-IDF
      3. Compute cosine similarity
      4. Deduplicate by title
      5. Rank by score
      6. Apply local-boost rule
      7. Return top N
    """
    start = time.time()

    # Step 1: Translate
    q_translated = translate_query(q)

    # Step 2: Vectorize
    q_vec = vectorizer.transform([q_translated])

    # Step 3: Cosine similarity
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    # Step 4: Rank
    df = df.copy()
    df["score"] = scores
    ranked = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Step 5: Deduplicate
    ranked = ranked.drop_duplicates(
        subset=["title"], keep="first"
    ).reset_index(drop=True)

    # Step 6: Top N
    top_results = ranked.head(top_n).copy()
    top_results["boosted"] = False

    # Step 7: Local-boost rule
    local_in_top = top_results[top_results["is_local"] == True]
    if len(local_in_top) == 0:
        local_products = ranked[ranked["is_local"] == True]
        if len(local_products) > 0:
            best_local = local_products.iloc[[0]]
            if best_local["score"].values[0] >= SIMILARITY_THRESHOLD:
                top_results = pd.concat(
                    [top_results.head(top_n - 1), best_local],
                    ignore_index=True
                )
                top_results["boosted"] = [False] * (top_n - 1) + [True]

    elapsed = (time.time() - start) * 1000
    top_results["query_time_ms"] = round(elapsed, 2)

    return top_results[[
        "sku", "title", "category", "material",
        "origin_district", "price_rwf", "score",
        "boosted", "query_time_ms"
    ]]

# ── Print Results ─────────────────────────────────────────────
def print_results(results, query):
    """Print results in clean readable terminal format."""
    print()
    print("=" * 60)
    print(f"  Query      : {query}")
    print(f"  Translated : {translate_query(query)}")
    print(f"  Query time : {results['query_time_ms'].iloc[0]} ms")
    print("=" * 60)
    for i, row in results.iterrows():
        boosted_tag = " *** LOCAL BOOST ***" if row["boosted"] else ""
        print(f"\n  Rank {i+1}{boosted_tag}")
        print(f"  SKU        : {row['sku']}")
        print(f"  Product    : {row['title']}")
        print(f"  Category   : {row['category']}")
        print(f"  Material   : {row['material']}")
        print(f"  District   : {row['origin_district']}")
        print(f"  Price(RWF) : {row['price_rwf']:,}")
        print(f"  Score      : {row['score']:.4f}")
        print("  " + "-" * 56)
    print()

# ── Main CLI ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Made in Rwanda Content Recommender — G5 S2.T1.3"
    )
    parser.add_argument(
        "--q",
        type=str,
        required=True,
        help="Search query in English, French, or Kinyarwanda"
    )
    args = parser.parse_args()

    df = load_catalog("catalog.csv")
    vectorizer, tfidf_matrix = build_index(df)
    results = recommend(args.q, df, vectorizer, tfidf_matrix)
    print_results(results, args.q)
