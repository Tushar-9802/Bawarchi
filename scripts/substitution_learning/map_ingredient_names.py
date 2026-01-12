"""
Ingredient Name Mapper
Maps detection class names to recipe ingredient variants using multi-strategy matching.
"""

import yaml
import json
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import re

def clean_ingredient_name(name):
    """Clean and normalize ingredient name"""
    # Remove content in parentheses and brackets
    name = re.sub(r'\([^)]*\)', '', name)
    name = re.sub(r'\[[^\]]*\]', '', name)
    # Remove special markers
    name = re.sub(r'-[^-]+-', ' ', name)  # Remove "-X-" patterns
    # Lowercase and strip
    name = name.lower().strip()
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name)
    return name

def extract_aliases(name):
    """Extract aliases from ingredient name"""
    aliases = []
    # Find content in dashes: "Chili Pepper -Khursani-"
    dash_pattern = r'-([^-]+)-'
    matches = re.findall(dash_pattern, name)
    aliases.extend([m.strip().lower() for m in matches])
    return aliases

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def similarity_score(s1, s2):
    """Calculate similarity score (0-1) based on Levenshtein distance"""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(s1, s2) / max_len)

def word_overlap_score(s1, s2):
    """Calculate word overlap score"""
    words1 = set(s1.split())
    words2 = set(s2.split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)

def match_ingredient(detection_class, recipe_ingredients_freq, min_freq=10):
    """
    Multi-strategy matching: exact → alias → substring → word overlap → Levenshtein
    Returns list of (recipe_ingredient, confidence, strategy) tuples
    """
    matches = []
    
    # Clean detection class name
    clean_name = clean_ingredient_name(detection_class)
    aliases = extract_aliases(detection_class)
    search_terms = [clean_name] + aliases
    
    # Strategy 1: Exact match
    for term in search_terms:
        for recipe_ing, freq in recipe_ingredients_freq.items():
            if freq < min_freq:
                continue
            if term == recipe_ing:
                matches.append((recipe_ing, 1.0, 'exact'))
    
    if matches:
        return matches
    
    # Strategy 2: Substring match (bidirectional)
    for term in search_terms:
        for recipe_ing, freq in recipe_ingredients_freq.items():
            if freq < min_freq:
                continue
            if term in recipe_ing or recipe_ing in term:
                # Calculate confidence based on length ratio
                len_ratio = min(len(term), len(recipe_ing)) / max(len(term), len(recipe_ing))
                matches.append((recipe_ing, len_ratio * 0.9, 'substring'))
    
    if matches:
        return matches
    
    # Strategy 3: Word overlap
    for term in search_terms:
        for recipe_ing, freq in recipe_ingredients_freq.items():
            if freq < min_freq:
                continue
            overlap = word_overlap_score(term, recipe_ing)
            if overlap > 0.5:
                matches.append((recipe_ing, overlap * 0.8, 'word_overlap'))
    
    if matches:
        return matches
    
    # Strategy 4: Levenshtein similarity (last resort)
    for term in search_terms:
        for recipe_ing, freq in recipe_ingredients_freq.items():
            if freq < min_freq:
                continue
            sim = similarity_score(term, recipe_ing)
            if sim > 0.75:
                matches.append((recipe_ing, sim * 0.7, 'levenshtein'))
    
    return matches

def load_recipe_ingredients(recipes_parquet_path):
    """Extract all ingredient strings from recipes"""
    print(f"Loading recipes from {recipes_parquet_path}")
    df = pd.read_parquet(recipes_parquet_path)
    
    # Collect all ingredients
    all_ingredients = []
    for ingredients_list in df['ingredients']:
        if isinstance(ingredients_list, list):
            all_ingredients.extend([ing.lower().strip() for ing in ingredients_list])
    
    # Count frequencies
    ingredient_freq = Counter(all_ingredients)
    
    print(f"Found {len(ingredient_freq)} unique ingredients")
    print(f"Total ingredient mentions: {sum(ingredient_freq.values())}")
    
    return dict(ingredient_freq)

def main():
    # Paths
    data_yaml = Path("data/merged/data.yaml")
    recipes_parquet = Path("data/recipes/recipes_processed.parquet")
    output_dir = Path("data/substitution")
    output_file = output_dir / "ingredient_mapping.json"
    unmapped_file = output_dir / "unmapped_classes.txt"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load detection classes
    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found")
        return
    
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    detection_classes = data['names']
    
    print(f"Loaded {len(detection_classes)} detection classes")
    
    # Load recipe ingredients
    if not recipes_parquet.exists():
        print(f"Error: {recipes_parquet} not found")
        return
    
    recipe_ingredients_freq = load_recipe_ingredients(recipes_parquet)
    
    # Map each detection class
    print("\nMapping detection classes to recipe ingredients...")
    mapping = {}
    unmapped = []
    low_confidence = []
    
    for i, det_class in enumerate(detection_classes):
        matches = match_ingredient(det_class, recipe_ingredients_freq, min_freq=10)
        
        if not matches:
            unmapped.append(det_class)
            print(f"  [{i+1}/{len(detection_classes)}] {det_class}: NO MATCH")
            continue
        
        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Take top matches (confidence > 0.6)
        good_matches = [m for m in matches if m[1] > 0.6]
        
        if not good_matches:
            unmapped.append(det_class)
            print(f"  [{i+1}/{len(detection_classes)}] {det_class}: LOW CONFIDENCE")
            continue
        
        # Store top matches (max 5)
        top_matches = good_matches[:5]
        mapping[det_class] = [m[0] for m in top_matches]
        
        # Flag low confidence if best match < 0.8
        if top_matches[0][1] < 0.8:
            low_confidence.append((det_class, top_matches))
        
        print(f"  [{i+1}/{len(detection_classes)}] {det_class}: {len(top_matches)} matches ({top_matches[0][2]})")
    
    # Save mapping
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\nMapping saved to {output_file}")
    print(f"Mapped: {len(mapping)}/{len(detection_classes)} classes")
    print(f"Unmapped: {len(unmapped)} classes")
    
    # Save unmapped classes
    if unmapped:
        with open(unmapped_file, 'w', encoding='utf-8') as f:
            f.write("Unmapped Detection Classes\n")
            f.write("=" * 50 + "\n\n")
            for det_class in unmapped:
                f.write(f"{det_class}\n")
        print(f"Unmapped classes saved to {unmapped_file}")
    
    # Print low confidence mappings for review
    if low_confidence:
        print("\nLow confidence mappings (manual review recommended):")
        for det_class, matches in low_confidence[:10]:
            print(f"  {det_class}:")
            for recipe_ing, conf, strategy in matches[:3]:
                print(f"    - {recipe_ing} (confidence: {conf:.2f}, strategy: {strategy})")
    
    # Statistics
    print("\nMapping statistics:")
    single_match = sum(1 for v in mapping.values() if len(v) == 1)
    multi_match = sum(1 for v in mapping.values() if len(v) > 1)
    print(f"  Single match: {single_match}")
    print(f"  Multiple matches: {multi_match}")
    print(f"  Coverage: {len(mapping)/len(detection_classes)*100:.1f}%")

if __name__ == "__main__":
    main()