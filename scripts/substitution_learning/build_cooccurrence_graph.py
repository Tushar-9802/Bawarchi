"""
Co-occurrence Graph Builder
Builds PMI (Pointwise Mutual Information) matrix from recipe corpus.
Tracks Indian vs Mexican co-occurrence separately.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import dok_matrix, save_npz
from collections import defaultdict
import yaml

def load_detection_classes(data_yaml_path):
    """Load detection class names"""
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['names']

def load_ingredient_mapping(mapping_path):
    """Load detection class to recipe ingredient mapping"""
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def reverse_mapping(ingredient_mapping):
    """Create recipe ingredient to detection class reverse mapping"""
    reverse = {}
    for det_class, recipe_ings in ingredient_mapping.items():
        for recipe_ing in recipe_ings:
            if recipe_ing not in reverse:
                reverse[recipe_ing] = []
            reverse[recipe_ing].append(det_class)
    return reverse

def detect_cuisine(recipe_row):
    """Detect if recipe is Indian, Mexican, or other"""
    title = str(recipe_row.get('Name', '')).lower()
    keywords = str(recipe_row.get('Keywords', '')).lower()
    
    indian_keywords = ['indian', 'curry', 'masala', 'tandoori', 'biryani', 'dal', 'paneer']
    mexican_keywords = ['mexican', 'taco', 'burrito', 'salsa', 'enchilada', 'tortilla', 'quesadilla']
    
    is_indian = any(kw in title or kw in keywords for kw in indian_keywords)
    is_mexican = any(kw in title or kw in keywords for kw in mexican_keywords)
    
    if is_indian and not is_mexican:
        return 'indian'
    elif is_mexican and not is_indian:
        return 'mexican'
    elif is_indian and is_mexican:
        return 'fusion'
    else:
        return 'other'

def build_cooccurrence_matrices(recipes_df, detection_classes, reverse_map, min_recipes=5):
    """
    Build co-occurrence matrices with PMI calculation
    Returns: overall matrix, indian matrix, mexican matrix
    """
    n_classes = len(detection_classes)
    class_to_idx = {cls: i for i, cls in enumerate(detection_classes)}
    
    # Initialize sparse matrices
    cooccur_overall = dok_matrix((n_classes, n_classes), dtype=np.float32)
    cooccur_indian = dok_matrix((n_classes, n_classes), dtype=np.float32)
    cooccur_mexican = dok_matrix((n_classes, n_classes), dtype=np.float32)
    
    # Count ingredient occurrences
    ingredient_counts_overall = defaultdict(int)
    ingredient_counts_indian = defaultdict(int)
    ingredient_counts_mexican = defaultdict(int)
    
    total_recipes = {'overall': 0, 'indian': 0, 'mexican': 0}
    
    print(f"Processing {len(recipes_df)} recipes...")
    
    for idx, row in recipes_df.iterrows():
        if idx % 5000 == 0:
            print(f"  Processed {idx}/{len(recipes_df)} recipes")
        
        # Get ingredients and map to detection classes
        ingredients_value = row.get('RecipeIngredientParts')
        ingredients = []
        
        # Check for null
        if ingredients_value is None:
            continue
        
        # Handle numpy array (most common)
        if hasattr(ingredients_value, '__array__'):
            ingredients = list(ingredients_value)
        elif isinstance(ingredients_value, list):
            ingredients = ingredients_value
        elif isinstance(ingredients_value, str):
            # Parse string representation
            try:
                import ast
                parsed = ast.literal_eval(ingredients_value)
                if isinstance(parsed, list):
                    ingredients = parsed
            except:
                # Try splitting by delimiters
                if ',' in ingredients_value:
                    ingredients = [x.strip() for x in ingredients_value.split(',')]
                elif ';' in ingredients_value:
                    ingredients = [x.strip() for x in ingredients_value.split(';')]
        
        if not ingredients:
            continue
        
        ingredients = [ing.lower().strip() for ing in ingredients if ing and isinstance(ing, str)]
        
        # Map to detection classes
        detected_classes = set()
        for ing in ingredients:
            if ing in reverse_map:
                detected_classes.update(reverse_map[ing])
        
        if len(detected_classes) < 2:
            continue
        
        # Detect cuisine
        cuisine = detect_cuisine(row)
        
        # Update counts
        detected_classes_list = list(detected_classes)
        
        # Overall
        total_recipes['overall'] += 1
        for cls in detected_classes_list:
            ingredient_counts_overall[cls] += 1
        
        # Cuisine-specific
        if cuisine == 'indian':
            total_recipes['indian'] += 1
            for cls in detected_classes_list:
                ingredient_counts_indian[cls] += 1
        elif cuisine == 'mexican':
            total_recipes['mexican'] += 1
            for cls in detected_classes_list:
                ingredient_counts_mexican[cls] += 1
        
        # Build co-occurrence pairs
        for i, cls1 in enumerate(detected_classes_list):
            for cls2 in detected_classes_list[i+1:]:
                idx1 = class_to_idx[cls1]
                idx2 = class_to_idx[cls2]
                
                # Overall
                cooccur_overall[idx1, idx2] += 1
                cooccur_overall[idx2, idx1] += 1
                
                # Cuisine-specific
                if cuisine == 'indian':
                    cooccur_indian[idx1, idx2] += 1
                    cooccur_indian[idx2, idx1] += 1
                elif cuisine == 'mexican':
                    cooccur_mexican[idx1, idx2] += 1
                    cooccur_mexican[idx2, idx1] += 1
    
    print(f"\nRecipe counts:")
    print(f"  Overall: {total_recipes['overall']}")
    print(f"  Indian: {total_recipes['indian']}")
    print(f"  Mexican: {total_recipes['mexican']}")
    
    # Calculate PMI
    print("\nCalculating PMI...")
    
    def calculate_pmi(cooccur_matrix, ingredient_counts, total_recipes_count, min_count=5):
        """Calculate PMI from co-occurrence matrix"""
        pmi_matrix = dok_matrix((n_classes, n_classes), dtype=np.float32)
        
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                cooccur_count = cooccur_matrix[i, j]
                
                if cooccur_count < min_count:
                    continue
                
                cls1 = detection_classes[i]
                cls2 = detection_classes[j]
                
                p_i = ingredient_counts.get(cls1, 0) / total_recipes_count
                p_j = ingredient_counts.get(cls2, 0) / total_recipes_count
                p_ij = cooccur_count / total_recipes_count
                
                if p_i > 0 and p_j > 0 and p_ij > 0:
                    pmi = np.log(p_ij / (p_i * p_j))
                    pmi_matrix[i, j] = pmi
                    pmi_matrix[j, i] = pmi
        
        return pmi_matrix
    
    pmi_overall = calculate_pmi(cooccur_overall, ingredient_counts_overall, total_recipes['overall'], min_recipes)
    pmi_indian = calculate_pmi(cooccur_indian, ingredient_counts_indian, total_recipes['indian'], min_recipes)
    pmi_mexican = calculate_pmi(cooccur_mexican, ingredient_counts_mexican, total_recipes['mexican'], min_recipes)
    
    return pmi_overall, pmi_indian, pmi_mexican, ingredient_counts_overall

def main():
    # Paths
    data_yaml = Path("data/merged/data.yaml")
    recipes_parquet = Path("data/recipes/food_com/recipes.parquet")
    mapping_file = Path("data/substitution/ingredient_mapping.json")
    output_dir = Path("data/substitution")
    
    # Check inputs
    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found")
        return
    if not recipes_parquet.exists():
        print(f"Error: {recipes_parquet} not found")
        return
    if not mapping_file.exists():
        print(f"Error: {mapping_file} not found. Run map_ingredient_names.py first")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    detection_classes = load_detection_classes(data_yaml)
    ingredient_mapping = load_ingredient_mapping(mapping_file)
    reverse_map = reverse_mapping(ingredient_mapping)
    
    print(f"Detection classes: {len(detection_classes)}")
    print(f"Mapped classes: {len(ingredient_mapping)}")
    print(f"Recipe ingredients mapped: {len(reverse_map)}")
    
    # Load recipes
    print(f"\nLoading recipes from {recipes_parquet}")
    recipes_df = pd.read_parquet(recipes_parquet)
    print(f"Loaded {len(recipes_df)} recipes")
    
    # Debug: Check ingredient format
    if len(recipes_df) > 0:
        sample = recipes_df['RecipeIngredientParts'].iloc[0]
        print(f"Sample ingredient format: type={type(sample).__name__}")
        if hasattr(sample, '__array__'):
            print(f"Detected as numpy array with {len(sample)} ingredients")
        else:
            print(f"Sample value={sample}")
    
    # Build matrices
    pmi_overall, pmi_indian, pmi_mexican, ingredient_counts = build_cooccurrence_matrices(
        recipes_df, detection_classes, reverse_map
    )
    
    # Save matrices
    print("\nSaving matrices...")
    save_npz(output_dir / "pmi_matrix.npz", pmi_overall.tocsr())
    save_npz(output_dir / "pmi_indian.npz", pmi_indian.tocsr())
    save_npz(output_dir / "pmi_mexican.npz", pmi_mexican.tocsr())
    
    # Save ingredient counts
    with open(output_dir / "ingredient_counts.json", 'w', encoding='utf-8') as f:
        json.dump(ingredient_counts, f, indent=2)
    
    # Statistics
    print("\nMatrix statistics:")
    print(f"  Overall non-zero entries: {pmi_overall.nnz}")
    print(f"  Indian non-zero entries: {pmi_indian.nnz}")
    print(f"  Mexican non-zero entries: {pmi_mexican.nnz}")
    
    # Find top co-occurring pairs
    print("\nTop 10 co-occurring pairs (overall):")
    pairs = []
    for i in range(len(detection_classes)):
        for j in range(i+1, len(detection_classes)):
            if pmi_overall[i, j] > 0:
                pairs.append((detection_classes[i], detection_classes[j], pmi_overall[i, j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    for cls1, cls2, pmi in pairs[:10]:
        print(f"  {cls1} <-> {cls2}: PMI = {pmi:.3f}")
    
    # Save statistics
    stats = {
        'total_recipes_processed': len(recipes_df),
        'detection_classes': len(detection_classes),
        'overall_pairs': pmi_overall.nnz,
        'indian_pairs': pmi_indian.nnz,
        'mexican_pairs': pmi_mexican.nnz,
        'top_pairs': [(cls1, cls2, float(pmi)) for cls1, cls2, pmi in pairs[:20]]
    }
    
    with open(output_dir / "cooccurrence_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nSaved matrices and statistics to {output_dir}/")

if __name__ == "__main__":
    main()