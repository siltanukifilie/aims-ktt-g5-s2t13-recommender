# Made in Rwanda Content Recommender
## AIMS KTT Fellowship Hackathon 2026 | G5 | S2.T1.3

---

## The Problem

Every day, hundreds of buyers in Rwanda and across Africa
search online for handmade leather bags, agaseke baskets,
beaded jewelry and kitenge dresses.

They never find the local artisan who makes exactly what they want.

Instead, the algorithm shows them Nike. Zara. Alibaba.

Local artisans lose sales not because their products are worse,
but because global e-commerce algorithms favor
high-volume international sellers.

This system fixes that.

---

## What This System Does

A content-based niche-first recommender that:
- Takes any search query in English, French or Kinyarwanda
- Finds the most relevant Made-in-Rwanda products
- Applies a local-boost rule to always surface local artisans
- Reaches offline artisans with no smartphone via weekly SMS
- Runs in under 10ms on a free CPU with no paid API needed

---

## Key Results

| Metric | Score |
|---|---|
| NDCG@5 | 0.9833 |
| Local Presence Rate | 100% |
| Query time | under 10ms |
| Languages supported | English, French, Kinyarwanda |
| Products in catalog | 400 Made-in-Rwanda products |
| Queries evaluated | 120 queries |

---

## How to Run in 2 Commands on Free Colab CPU

Command 1 - Install dependencies:

    pip install pandas numpy scikit-learn

Command 2 - Run the recommender:

    python recommender.py --q "leather boots"

More example queries:

    python recommender.py --q "cadeau en cuir pour femme"
    python recommender.py --q "agaseke na impano"

---

## Repository Structure

    aims-ktt-g5-s2t13-recommender/
    |
    |-- recommender.py       : Main CLI recommender TF-IDF + local-boost
    |-- evaluate.py          : Evaluation NDCG@5 + Local Presence Rate
    |-- data_generator.py    : Generates all 3 synthetic datasets
    |
    |-- catalog.csv          : 400 Made-in-Rwanda products
    |-- queries.csv          : 120 search queries EN/FR/mixed
    |-- click_log.csv        : 5000 click events
    |
    |-- dispatcher.md        : Offline artisan weekly leads workflow
    |-- process_log.md       : Hour-by-hour log and LLM declarations
    |-- SIGNED.md            : Honor code
    |-- LICENSE              : MIT License
    |-- README.md            : This file

---

## How the Recommender Works

Step 1 - Query Translation
French and Kinyarwanda queries are translated to English keywords.

    cadeau en cuir pour femme
    becomes: gift leather leather leather woman leather bag

Step 2 - TF-IDF Index
A TF-IDF matrix is built on product title, description,
category, material and origin district.

Step 3 - Cosine Similarity
Each query vector is compared against all 400 product vectors.
Products are ranked by similarity score.

Step 4 - Local-Boost Rule
If no local product appears in top 5 results,
the best local match is injected as fallback.

Step 5 - Deduplication
Duplicate product titles are removed.
Only unique products appear in results.

---

## Example Output

    Query      : cadeau en cuir pour femme
    Translated : gift leather leather leather woman leather bag
    Query time : 5.08 ms

    Rank 1
    Product    : Leather Wallet Rwanda
    Category   : leather
    District   : Nyamirambo
    Price(RWF) : 82,612
    Score      : 0.4511

    Rank 2
    Product    : Leather Sandals Rwanda
    Category   : leather
    District   : Nyamirambo
    Price(RWF) : 135,569
    Score      : 0.4456

    Rank 3
    Product    : Leather Backpack Kigali
    Category   : leather
    District   : Nyamirambo
    Price(RWF) : 146,127
    Score      : 0.4329

---

## The Offline Artisan Problem

A leatherworker in Nyamirambo has no smartphone.
He cannot see who is searching for his products.

Our system solves this through a weekly SMS workflow:

    SMS in Kinyarwanda:
    Muraho! Iki cyumweru abantu 47 bashakaga
    inkweto na imifuko y uruhu i Kigali.
    Ibicuruzwa byawe byabonekwe 23 inshuro.
    Agent azaza Wagatanu saa tatu. 0788-300-100

    Translation:
    Hello! This week 47 people searched for leather
    sandals and bags in Kigali. Your products matched
    23 times. Agent comes Friday 9am.

See dispatcher.md for the full weekly workflow and economics.

---

## 3-Month Pilot Economics

| Metric | Value |
|---|---|
| Artisans in pilot | 20 |
| Total cost over 3 months | $1,208 |
| Cost per lead | $0.02 |
| Cost per artisan onboarded | $54 |
| Break-even GMV | $12,080 |
| Break-even month | Month 2 |
| Sales needed per artisan per month | 4 to 5 |

---

## Real Made-in-Rwanda Brands This Helps

Gahaya Links
Women's basketry cooperative.
Handweaves traditional agaseke peace baskets.
Exports globally but invisible on local search.

Inzuki Designs
Kigali-based jewelry cooperative.
Beaded necklaces, earrings and bracelets.
Run entirely by women artisans.

Crochet by Peace
Handmade fashion cooperative.
Kitenge dresses, blazers and accessories.
Employs over 30 women artisans.

---

## Constraints Addressed

| Constraint | Solution |
|---|---|
| No smartphone | SMS on basic feature phone |
| No internet | System runs on server |
| Illiteracy | Agent reads SMS aloud on Friday visit |
| Multiple languages | English, French, Kinyarwanda supported |
| Intermittent power | SMS stored on SIM card |
| No bank account | Agent collects cash on delivery |
| Low bandwidth | TF-IDF runs CPU-only |

---

## Why TF-IDF Over Sentence Embeddings

| Factor | TF-IDF | Sentence Embeddings |
|---|---|---|
| CPU-only | Yes | Needs GPU |
| Query time | Under 10ms | 200ms or more |
| Paid API | None needed | Often required |
| Interpretable | Easy to debug | Black box |
| Multilingual | Yes with dictionary | Native |
| Semantic queries | Misses them | Handles them |

Known weakness:
TF-IDF fails on purely semantic queries like
"something nice for my mother" where no keyword
directly matches product descriptions.

---

## Data Generation

All datasets are fully synthetic and reproducible.
Run this to regenerate all data:

    python data_generator.py

Generates:
- catalog.csv    : 400 products across 5 categories
- queries.csv    : 120 queries in English, French, mixed
- click_log.csv  : 5000 click events with position-bias

Seed: random.seed(42) ensures full reproducibility.

---

## Model on Hugging Face

TF-IDF vectorizer and product index:
Link to be added after upload

---

## 4-Minute Video

Full solution walkthrough:
Link to be added after recording

---

## Submission Checklist

- recommender.py with CLI
- evaluate.py with NDCG@5 and Local Presence Rate
- data_generator.py reproducible datasets
- dispatcher.md offline artisan workflow
- process_log.md LLM declarations
- SIGNED.md honor code
- LICENSE MIT
- HuggingFace model link - to be added
- 4-minute video link - to be added

---

## License

MIT License — see LICENSE file for details.
