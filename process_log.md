# Process Log — G5 | S2.T1.3 | Made in Rwanda Content Recommender
## AIMS KTT Fellowship Hackathon 2026

---

## Declared Tools

| Tool | Version | Purpose |
|---|---|---|
| Claude | Anthropic claude-sonnet-4-6 | Strategy, code structure, documentation guidance |
| Google Colab | Free CPU | Running and testing all code |
| GitHub | Latest | Version control and code hosting |

---

## Hour-by-Hour Timeline

### 09:30 — Rubric Walkthrough
- Attended AIMS KTT rubric walkthrough on Zoom
- Understood scoring criteria:
  20% Technical Quality, 20% Model Performance,
  20% Product and Business, 15% Communication,
  15% Data Handling, 10% Innovation
- Noted that Product and Business is the KTT signature criterion

### 10:00 — Received Challenge Brief
- Read full S2.T1.3 brief end to end
- Challenge: Made in Rwanda Content Recommender

### 10:00 to 10:30 — Research and Problem Understanding
- Searched and verified this is a real problem in Rwanda
- Local artisans in Nyamirambo are genuinely invisible
  on global e-commerce platforms like Jumia and Amazon
- Global algorithms prioritize high-volume international
  sellers over local Rwandan artisans
- Confirmed real Made-in-Rwanda brands affected:
  Gahaya Links, Inzuki Designs, Crochet by Peace
- Understood the offline artisan constraint:
  leatherworker in Nyamirambo with no smartphone
- Planned full architecture:
  TF-IDF + local-boost + SMS dispatcher workflow

### 10:30 to 11:00 — Set Up GitHub Repository
- Created public repo: aims-ktt-g5-s2t13-recommender
- Added SIGNED.md with honor code signed
- Added LICENSE MIT
- Added empty README.md and process_log.md
- Set up Google Colab connected to GitHub

### 11:00 to 11:45 — Built data_generator.py
- Generated catalog.csv: 400 Made-in-Rwanda products
  across 5 categories: apparel, leather, basketry,
  jewellery, home-decor
- Generated queries.csv: 120 queries in EN/FR/mixed
  including common misspellings
- Generated click_log.csv: 5,000 click events
  with position-bias noise model
- Key decision: anchored all leather products to
  Nyamirambo district as per brief requirement
- Verified all 3 files generated correctly

### 11:45 to 13:00 — Built recommender.py
- Implemented TF-IDF vectorizer on combined product text:
  title + description + category + material + district
- Implemented cosine similarity ranking against all 400 products
- Added local-boost rule: if no local product in top 5,
  inject best local match as fallback
- Added French to English translation dictionary
  to handle French queries like cadeau en cuir pour femme
- Added Kinyarwanda to English translation dictionary
  to handle queries like agaseke na impano
- Added deduplication to show only unique products
- Tested all 3 languages successfully
- Query time under 10ms on free Colab CPU

### 13:00 to 13:30 — Lunch Break

### 13:30 to 14:00 — Built evaluate.py
- Implemented NDCG@5 metric to measure ranking quality
- Implemented Local Presence Rate to measure
  how often local products appear in top 3
- Ran evaluation across all 120 queries
- Results:
  NDCG@5              = 0.9833
  Local Presence Rate = 100.0%
- Breakdown by language:
  English : NDCG@5 = 1.0000, Local Presence = 100%
  French  : NDCG@5 = 1.0000, Local Presence = 100%
  Mixed   : NDCG@5 = 0.9500, Local Presence = 100%

### 14:00 to 14:30 — Wrote dispatcher.md
- Designed full weekly leads workflow for offline artisan
- Defined 4 actors: artisan, cooperative agent,
  recommender system, MTN SMS gateway
- Wrote SMS digest in Kinyarwanda with English translation
- Designed step-by-step workflow:
  Monday system runs, Tuesday agent reviews,
  Wednesday SMS sent, Thursday artisan prepares,
  Friday agent visits, Friday afternoon sales attempted
- Proposed 3-month pilot with 20 artisans across Kigali
- Calculated unit economics:
  Cost per lead: $0.02
  Cost per artisan onboarded: $54 over 3 months
  Break-even GMV: $12,080
  Break-even month: Month 2
- Addressed all real constraints:
  no smartphone, illiteracy, multiple languages,
  intermittent power, no bank account, low bandwidth

### 14:30 to 15:00 — Wrote README.md and process_log.md
- Full project documentation in README.md
- How to run in 2 commands on free Colab CPU
- Example outputs, repo structure, business context
- Unit economics and pilot summary
- Constraints addressed table
- TF-IDF vs sentence embeddings comparison table

### 15:00 to 15:30 — Final Push and Verification
- Pushed all files to GitHub
- Verified all links open in incognito browser
- Verified README runs in 2 commands on fresh Colab
- Verified all required files present:
  recommender.py, evaluate.py, data_generator.py,
  catalog.csv, queries.csv, click_log.csv,
  dispatcher.md, process_log.md, SIGNED.md, LICENSE

### 15:30 to 16:00 — Video Recording
- Practiced video once with timer
- Recorded 4-minute video following exact structure:
  0:00 to 0:30 — On-camera intro with scores
  0:30 to 1:30 — Live code walk of recommend() function
  1:30 to 2:30 — Live demo of English and French queries
  2:30 to 3:30 — Walked dispatcher.md on screen
  3:30 to 4:00 — Answered 3 spoken questions unscripted
- Uploaded video to YouTube unlisted
- Added video URL to README.md

### 15:45 — Submitted Everything
- Submitted GitHub URL
- Submitted HuggingFace model link
- Submitted 4-minute video URL
- Verified all links work in incognito browser
- Submission complete before 4:00 PM deadline

---

## Three Sample Prompts Sent to Claude

### Prompt 1:
"I have an AIMS KTT hackathon challenge S2.T1.3
Made in Rwanda Content Recommender. Help me understand
what I need to build and plan my approach."

### Prompt 2:
"The French query cadeau en cuir pour femme is returning
wrong results with score 0.0000. How do I fix the
TF-IDF recommender to handle French queries correctly?"

### Prompt 3:
"Write dispatcher.md for an offline artisan in Nyamirambo
with no smartphone. Include weekly SMS workflow in
Kinyarwanda, numbers, 3-month pilot with 20 artisans
and unit economics with break-even analysis."

---

## One Prompt Discarded

### Discarded Prompt:
"Build me a complete RAG system with sentence embeddings
for the Made in Rwanda recommender."

### Why Discarded:
The brief requires CPU-only with query time under 250ms.
Sentence embeddings need GPU or heavy models and would
exceed the time constraint on a free Colab CPU.
TF-IDF is faster, interpretable and meets all requirements.
A clean simple correct baseline beats a half-working
production solution.

---

## Hardest Decision

The hardest decision was choosing between TF-IDF and
sentence embeddings for the recommender engine.

Sentence embeddings would handle semantic queries better.
For example "gift for my mother" would match leather bags
even without the word leather appearing in the query.
But sentence embeddings require a GPU or paid API and
exceed the 250ms query time constraint on free Colab CPU.

TF-IDF is fast, interpretable, runs on any CPU in under
10ms, and requires no paid API. The weakness is that it
misses purely semantic queries. This is an honest
trade-off made deliberately given the constraints.

The addition of French and Kinyarwanda translation
dictionaries compensated significantly for the keyword
matching limitation and allowed the system to serve
multilingual users across Rwanda effectively.
