"""
data_generator.py
=================
AIMS KTT Hackathon 2026 | G5 | S2.T1.3
Made in Rwanda Content Recommender

Generates three synthetic datasets:
  - catalog.csv    : 400 Made-in-Rwanda products across 5 categories
  - queries.csv    : 120 search queries (EN / FR / code-switched + misspellings)
  - click_log.csv  : 5,000 click events with position-bias noise model

Design decisions:
  - Leather products anchored to Nyamirambo district (brief requirement)
  - Other categories spread across 7 Rwanda districts
  - random.seed(42) ensures full reproducibility

Author : Sltanu Kifile Alemu
Date   : 22 April 2026
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# Districts
leather_district = "Nyamirambo"
other_districts = [
    "Kimironko", "Kicukiro", "Gasabo",
    "Musanze", "Huye", "Rubavu", "Muhanga"
]

# Product Data
catalog_data = {
    "apparel": {
        "materials": ["cotton", "linen", "silk", "wool", "kitenge"],
        "products": [
            ("Rwandan Kitenge Dress", "Beautiful handmade kitenge dress with traditional African patterns made by local artisans in Kigali"),
            ("Cotton Wrap Skirt", "Handwoven cotton wrap skirt made by local artisans in Rwanda cooperative"),
            ("Linen Safari Shirt", "Lightweight linen shirt perfect for Rwandan climate handmade locally"),
            ("Imigongo Print Shirt", "Shirt featuring traditional Rwandan Imigongo geometric art patterns"),
            ("Handmade Wool Sweater", "Warm wool sweater knitted by Kigali women cooperative artisans"),
            ("Kitenge Blazer", "Professional blazer made from authentic African kitenge fabric Rwanda"),
            ("Cotton Summer Dress", "Light cotton dress handmade by local Rwandan seamstress cooperative"),
            ("African Print Jumpsuit", "Trendy jumpsuit made from local African print fabric Kigali artisan"),
        ]
    },
    "leather": {
        "materials": ["leather", "suede", "cowhide"],
        "products": [
            ("Kigali Leather Boots", "Handcrafted leather boots made by skilled Nyamirambo leatherworker artisans"),
            ("Rwanda Leather Handbag", "Premium leather handbag crafted by local Nyamirambo leatherworkers"),
            ("Leather Wallet Rwanda", "Genuine cowhide wallet handmade by Nyamirambo leather artisan"),
            ("Leather Belt Kigali", "Hand-stitched leather belt from Rwandan cowhide by Nyamirambo artisan"),
            ("Suede Loafers Kigali", "Comfortable suede loafers handcrafted in Nyamirambo by skilled cobbler"),
            ("Leather Sandals Rwanda", "Traditional leather sandals handmade in Nyamirambo workshop"),
            ("Leather Backpack Kigali", "Durable leather backpack handcrafted by Nyamirambo local artisans"),
            ("Cowhide Purse Rwanda", "Beautiful cowhide purse made by skilled Nyamirambo leatherworkers"),
        ]
    },
    "basketry": {
        "materials": ["sisal", "raffia", "bamboo", "papyrus"],
        "products": [
            ("Rwandan Peace Basket", "Traditional Rwandan agaseke peace basket handwoven by women cooperative"),
            ("Sisal Shopping Basket", "Eco-friendly sisal shopping basket woven in rural Rwanda cooperative"),
            ("Papyrus Storage Basket", "Large papyrus storage basket from Lake Kivu region artisans"),
            ("Bamboo Fruit Basket", "Beautiful bamboo fruit basket handwoven by Musanze artisans cooperative"),
            ("Raffia Gift Basket", "Decorative raffia gift basket with traditional Rwandan patterns"),
            ("Woven Market Basket", "Sturdy woven market basket from Huye women cooperative"),
            ("Mini Agaseke Basket", "Small traditional agaseke basket perfect as Made in Rwanda gift"),
            ("Sisal Wall Decoration", "Decorative sisal wall piece handwoven by Kimironko artisans"),
        ]
    },
    "jewellery": {
        "materials": ["beads", "silver", "copper", "bone", "wood"],
        "products": [
            ("Rwandan Beaded Necklace", "Colorful beaded necklace handmade by Kigali women cooperative artisans"),
            ("Copper Bangle Rwanda", "Hand-beaten copper bangle made by local Rwandan artisans"),
            ("Silver Earrings Kigali", "Elegant silver earrings crafted by skilled Rwandan jewelers cooperative"),
            ("Bone Bracelet Rwanda", "Traditional bone bracelet handcarved by skilled Rwanda artisan"),
            ("Wooden Bead Necklace", "Natural wooden bead necklace from sustainable Rwandan forest materials"),
            ("Beaded Anklet Kigali", "Colorful beaded anklet handmade in Kigali cooperative workshop"),
            ("Copper Ring Rwanda", "Handcrafted copper ring made by skilled Kigali artisans"),
            ("Beaded Hair Pins", "Traditional beaded hair pins made by Rwandan women cooperative"),
        ]
    },
    "home-decor": {
        "materials": ["wood", "clay", "sisal", "papyrus", "recycled"],
        "products": [
            ("Imigongo Wall Art", "Traditional Rwandan Imigongo geometric wall art handpainted by local artist"),
            ("Clay Cooking Pot Rwanda", "Traditional Rwandan clay cooking pot handmade by Huye potters cooperative"),
            ("Wooden Serving Bowl", "Hand-carved wooden serving bowl from Rwandan mahogany by local artisan"),
            ("Sisal Table Mat", "Eco-friendly sisal table mat woven by local Rwandan cooperative"),
            ("Papyrus Lampshade", "Beautiful papyrus lampshade handwoven by Rubavu artisans"),
            ("Recycled Metal Sculpture", "Artistic metal sculpture made from recycled materials by Kigali artisan"),
            ("Clay Tea Set Rwanda", "Traditional clay tea set handmade by skilled Huye potters"),
            ("Wooden Photo Frame", "Hand-carved wooden photo frame from sustainable Rwandan timber"),
        ]
    }
}

# Generate 400 Products
rows = []
sku_counter = 1000
artisan_counter = 1

for category, data in catalog_data.items():
    products = data["products"]
    materials = data["materials"]
    for i in range(80):
        base = products[i % len(products)]
        district = leather_district if category == "leather" else other_districts[i % len(other_districts)]
        rows.append({
            "sku"            : f"RW-{sku_counter}",
            "title"          : base[0],
            "description"    : base[1],
            "category"       : category,
            "material"       : materials[i % len(materials)],
            "origin_district": district,
            "price_rwf"      : random.randint(2000, 150000),
            "artisan_id"     : f"ART-{artisan_counter:03d}"
        })
        sku_counter += 1
        artisan_counter = (artisan_counter % 50) + 1

catalog_df = pd.DataFrame(rows)
catalog_df.to_csv("catalog.csv", index=False)

# Generate 120 Queries
queries_en = [
    "leather boots", "handmade basket", "African dress", "beaded necklace",
    "wooden bowl", "wall art", "leather bag", "cotton shirt", "silver earrings",
    "traditional pottery", "woven mat", "copper bracelet", "kitenge fabric",
    "peace basket", "leather sandals", "African jewelry", "handmade purse",
    "bamboo basket", "Rwandan art", "local crafts", "leather wallet",
    "African print dress", "handwoven basket", "clay pot", "wooden sculpture",
    "beaded bracelet", "traditional basket", "leather handbag", "African necklace",
    "handmade jewelry", "sisal basket", "Rwandan dress", "leather belt",
    "wooden frame", "traditional earrings", "cotton dress", "raffia basket",
    "Rwandan jewelry", "handmade bag", "African basket"
]
queries_fr = [
    "bottes en cuir", "sac en cuir", "robe africaine", "collier en perles",
    "bol en bois", "art mural", "panier tresse", "chemise en coton",
    "boucles d oreilles", "poterie traditionnelle", "natte tissee",
    "bracelet en cuivre", "tissu kitenge", "panier de paix",
    "sandales en cuir", "bijoux africains", "sac a main", "panier en bambou",
    "art rwandais", "artisanat local", "portefeuille en cuir",
    "robe imprimee africaine", "panier artisanal", "pot en argile",
    "sculpture en bois", "bracelet perle", "panier traditionnel",
    "sac en cuir fait main", "collier africain", "bijoux faits main",
    "cadeau en cuir pour femme", "cadeau artisanal rwanda",
    "bijoux rwandais", "sac fait main", "panier africain",
    "robe rwandaise", "ceinture en cuir", "cadre en bois",
    "boucles traditionnelles", "robe en coton"
]
queries_mixed = [
    "leather bag Kigali", "agaseke basket Rwanda", "imigongo art Rwanda",
    "kitenge dress Kigali", "Rwanda handmade jewelry", "local leather boots Rwanda",
    "made in Rwanda basket", "Kigali artisan bag", "Rwanda traditional art",
    "handmade Rwanda gift", "leather sandal Nyamirambo", "Rwanda beaded necklace",
    "Kigali wooden bowl", "Rwanda clay pot", "handcrafted Rwanda",
    "local artisan Rwanda", "Rwanda copper bracelet", "Kigali fashion",
    "Rwanda eco basket", "artisanat Kigali", "sac Kigali leather",
    "Rwanda sisal mat", "bijoux Kigali", "panier Rwanda artisan",
    "robe Kigali local", "Rwanda gift basket", "Kigali traditional craft",
    "Rwanda women cooperative", "agaseke Kigali", "Rwanda sustainable fashion",
    "lether boots", "handmde basket", "afican dress", "beeded necklace",
    "wooven basket", "claay pot", "leathr bag", "africn jewelry",
    "rwandan bascket", "kitenge cloths"
]

all_queries = (queries_en + queries_fr + queries_mixed)[:120]
languages   = (["en"]*40 + ["fr"]*40 + ["mixed"]*40)[:120]

queries_df = pd.DataFrame({
    "query_id"         : [f"Q{i+1:03d}" for i in range(120)],
    "query_text"       : all_queries,
    "language"         : languages,
    "global_best_match": ["Global Brand Product"] * 120
})
queries_df.to_csv("queries.csv", index=False)

# Generate 5,000 Click Events
skus = catalog_df["sku"].tolist()
qids = queries_df["query_id"].tolist()
click_rows = []
for i in range(5000):
    position = np.random.choice(
        [1,2,3,4,5,6,7,8,9,10],
        p=[0.35,0.25,0.15,0.08,0.06,0.04,0.03,0.02,0.01,0.01]
    )
    click_rows.append({
        "event_id"   : f"EVT-{i+1:05d}",
        "query_id"   : random.choice(qids),
        "sku_clicked": random.choice(skus),
        "position"   : position,
        "clicked"    : 1
    })
click_df = pd.DataFrame(click_rows)
click_df.to_csv("click_log.csv", index=False)
