#!/usr/bin/env python3
"""
BAWARCHI - Recipe Processing Script
Extracts Food.com recipes, filters Indian/Mexican for T5 training
"""

import zipfile
from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path("C:/Users/Tushar/Desktop/bawarchi")
RECIPES_DIR = BASE_DIR / "data" / "recipes" / "food_com"
PROCESSED_DIR = BASE_DIR / "data" / "recipes" / "processed"

print("=" * 70)
print("BAWARCHI - RECIPE PROCESSING")
print("=" * 70)

# Step 1: Extract recipes.parquet.zip if needed
print("\n[STEP 1] EXTRACTING RECIPES")
print("-" * 70)

recipes_zip = RECIPES_DIR / "recipes.parquet.zip"
recipes_parquet = RECIPES_DIR / "recipes.parquet"

if recipes_parquet.exists():
    print(f"✓ recipes.parquet already extracted")
elif recipes_zip.exists():
    print(f"Extracting {recipes_zip.name}...")
    with zipfile.ZipFile(recipes_zip, 'r') as zip_ref:
        zip_ref.extractall(RECIPES_DIR)
    print(f"✓ Extracted to {RECIPES_DIR}")
else:
    print("✗ recipes.parquet.zip not found!")
    print(f"  Expected location: {recipes_zip}")
    exit(1)

# Step 2: Load recipes
print("\n[STEP 2] LOADING RECIPES")
print("-" * 70)

print("Loading Food.com recipes...")
df = pd.read_parquet(recipes_parquet)
print(f"✓ Loaded {len(df):,} recipes")

# Show sample
sample = df.iloc[0]
print(f"\n[SAMPLE RECIPE]")
print(f"Name: {sample['Name']}")
print(f"Ingredients: {sample.get('RecipeIngredientParts', 'N/A')[:3] if isinstance(sample.get('RecipeIngredientParts'), list) else 'N/A'}...")
print(f"Instructions: Available = {isinstance(sample.get('RecipeInstructions'), list)}")

# Step 3: Filter Indian recipes
print("\n[STEP 3] FILTERING INDIAN RECIPES")
print("-" * 70)

indian_keywords = [
    'curry', 'tikka', 'masala', 'biryani', 'tandoori', 'korma', 'vindaloo',
    'paneer', 'dal', 'daal', 'samosa', 'naan', 'roti', 'chapati', 'dosa',
    'idli', 'chutney', 'raita', 'lassi', 'chai', 'kheer', 'gulab jamun',
    'pakora', 'bhaji', 'aloo', 'gobi', 'palak', 'saag', 'jeera', 'garam',
    'turmeric', 'cumin', 'coriander', 'cardamom', 'ghee', 'indian',
    'basmati', 'mulligatawny', 'korma', 'madras', 'jalfrezi', 'passanda'
]

# Filter by name and description
print("Searching recipes with Indian keywords...")
indian_pattern = '|'.join(indian_keywords)
indian_mask = df['Name'].str.lower().str.contains(indian_pattern, na=False, regex=True)

if 'Description' in df.columns:
    indian_mask |= df['Description'].str.lower().str.contains(indian_pattern, na=False, regex=True)

indian_df = df[indian_mask].copy()
print(f"✓ Found {len(indian_df):,} Indian recipes ({len(indian_df)/len(df)*100:.1f}%)")

# Step 4: Filter Mexican recipes
print("\n[STEP 4] FILTERING MEXICAN RECIPES")
print("-" * 70)

mexican_keywords = [
    'taco', 'burrito', 'quesadilla', 'enchilada', 'tamale', 'tortilla',
    'salsa', 'guacamole', 'nachos', 'fajita', 'chimichanga', 'carnitas',
    'carne asada', 'chile', 'jalapeño', 'jalapeno', 'poblano', 'chipotle',
    'cilantro', 'queso', 'refried', 'mexican', 'tex-mex', 'tostada',
    'pozole', 'mole', 'chorizo', 'elote', 'pico de gallo', 'verde',
    'roja', 'oaxaca', 'yucatan', 'sonora', 'baja'
]

print("Searching recipes with Mexican keywords...")
mexican_pattern = '|'.join(mexican_keywords)
mexican_mask = df['Name'].str.lower().str.contains(mexican_pattern, na=False, regex=True)

if 'Description' in df.columns:
    mexican_mask |= df['Description'].str.lower().str.contains(mexican_pattern, na=False, regex=True)

mexican_df = df[mexican_mask].copy()
print(f"✓ Found {len(mexican_df):,} Mexican recipes ({len(mexican_df)/len(df)*100:.1f}%)")

# Step 5: Create fusion corpus
print("\n[STEP 5] CREATING FUSION CORPUS")
print("-" * 70)

indian_ids = set(indian_df['RecipeId'])
mexican_ids = set(mexican_df['RecipeId'])
overlap_ids = indian_ids & mexican_ids

print(f"Overlap (fusion recipes): {len(overlap_ids):,}")

# Combined corpus for T5 training
fusion_ids = indian_ids | mexican_ids
fusion_df = df[df['RecipeId'].isin(fusion_ids)].copy()

print(f"✓ Total fusion corpus: {len(fusion_df):,} recipes")
print(f"  Indian: {len(indian_ids):,}")
print(f"  Mexican: {len(mexican_ids):,}")
print(f"  Both: {len(overlap_ids):,}")

# Step 6: Save processed datasets
print("\n[STEP 6] SAVING PROCESSED DATASETS")
print("-" * 70)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

indian_df.to_parquet(PROCESSED_DIR / "indian_recipes.parquet")
print(f"✓ Saved: {PROCESSED_DIR / 'indian_recipes.parquet'}")

mexican_df.to_parquet(PROCESSED_DIR / "mexican_recipes.parquet")
print(f"✓ Saved: {PROCESSED_DIR / 'mexican_recipes.parquet'}")

fusion_df.to_parquet(PROCESSED_DIR / "fusion_recipes.parquet")
print(f"✓ Saved: {PROCESSED_DIR / 'fusion_recipes.parquet'}")

# Step 7: Sample recipes
print("\n[STEP 7] SAMPLE RECIPES")
print("-" * 70)

print("\n[INDIAN SAMPLE]")
indian_sample = indian_df.iloc[0]
print(f"Name: {indian_sample['Name']}")
print(f"Category: {indian_sample.get('RecipeCategory', 'N/A')}")

print("\n[MEXICAN SAMPLE]")
mexican_sample = mexican_df.iloc[0]
print(f"Name: {mexican_sample['Name']}")
print(f"Category: {mexican_sample.get('RecipeCategory', 'N/A')}")

if len(overlap_ids) > 0:
    print("\n[FUSION SAMPLE]")
    fusion_sample = fusion_df[fusion_df['RecipeId'].isin(overlap_ids)].iloc[0]
    print(f"Name: {fusion_sample['Name']}")
    print(f"Category: {fusion_sample.get('RecipeCategory', 'N/A')}")

# Summary
print("\n" + "=" * 70)
print("RECIPE PROCESSING COMPLETE")
print("=" * 70)

print(f"\n[FINAL STATS]")
print(f"Total recipes processed: {len(df):,}")
print(f"Indian recipes: {len(indian_df):,}")
print(f"Mexican recipes: {len(mexican_df):,}")
print(f"Fusion corpus: {len(fusion_df):,}")
print(f"\nProcessed files location: {PROCESSED_DIR}")

print("\n" + "=" * 70)