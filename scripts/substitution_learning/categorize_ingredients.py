"""
Ingredient Categorization Helper
Assists in manually categorizing 124 ingredients into culinary categories.
Uses embedding-based suggestions to speed up process.
"""

import yaml
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Category definitions with example keywords
CATEGORIES = {
    'protein': ['chicken', 'meat', 'fish', 'egg', 'paneer', 'tofu', 'lentil', 'dal', 'bean'],
    'vegetable': ['tomato', 'onion', 'potato', 'spinach', 'carrot', 'pepper', 'gourd', 'cabbage'],
    'spice': ['cumin', 'turmeric', 'chili', 'pepper', 'coriander', 'garam', 'masala', 'cinnamon'],
    'grain': ['rice', 'wheat', 'flour', 'bread', 'roti', 'tortilla', 'corn', 'maize'],
    'dairy': ['milk', 'cheese', 'yogurt', 'cream', 'butter', 'ghee', 'paneer'],
    'fruit': ['apple', 'banana', 'mango', 'orange', 'lemon', 'lime', 'berry', 'melon'],
    'condiment': ['sauce', 'paste', 'oil', 'vinegar', 'soy', 'ketchup', 'chutney'],
    'fat': ['oil', 'butter', 'ghee', 'lard', 'coconut oil']
}

def load_ingredient_names(data_yaml_path):
    """Load ingredient class names from data.yaml"""
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['names']

def suggest_category(ingredient_name, model):
    """Suggest category based on embedding similarity to keywords"""
    ingredient_name_clean = ingredient_name.lower().replace('-', ' ')
    
    # Simple keyword matching first
    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if keyword in ingredient_name_clean:
                return category
    
    # Fallback: embedding similarity to category keywords
    ingredient_embedding = model.encode([ingredient_name_clean])[0]
    
    best_category = None
    best_score = -1
    
    for category, keywords in CATEGORIES.items():
        keyword_embeddings = model.encode(keywords)
        # Average similarity to all keywords in category
        similarities = []
        for kw_emb in keyword_embeddings:
            sim = (ingredient_embedding @ kw_emb) / (
                (ingredient_embedding @ ingredient_embedding) ** 0.5 * 
                (kw_emb @ kw_emb) ** 0.5
            )
            similarities.append(sim)
        avg_sim = sum(similarities) / len(similarities)
        
        if avg_sim > best_score:
            best_score = avg_sim
            best_category = category
    
    return best_category

def interactive_categorization(ingredients, model, output_path):
    """Interactive categorization with user confirmation"""
    print(f"\nCategorizing {len(ingredients)} ingredients")
    print("Categories: protein, vegetable, spice, grain, dairy, fruit, condiment, fat")
    print("Commands: [Enter] = accept suggestion, [category] = override, 'q' = quit\n")
    
    categories = {}
    
    # Check if partial results exist
    if output_path.exists():
        print(f"Loading existing categorization from {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        print(f"Loaded {len(categories)} existing categories\n")
    
    for i, ingredient in enumerate(ingredients):
        # Skip if already categorized
        if ingredient in categories:
            continue
        
        # Get suggestion
        suggestion = suggest_category(ingredient, model)
        
        # Display
        print(f"[{i+1}/{len(ingredients)}] {ingredient}")
        print(f"  Suggested: {suggestion}")
        
        # User input
        user_input = input("  Category: ").strip().lower()
        
        if user_input == 'q':
            print("Quitting. Progress saved.")
            break
        elif user_input == '':
            categories[ingredient] = suggestion
        elif user_input in CATEGORIES:
            categories[ingredient] = user_input
        else:
            print(f"  Invalid category. Using suggestion: {suggestion}")
            categories[ingredient] = suggestion
        
        # Save progress every 10 items
        if (i + 1) % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(categories, f, indent=2)
    
    # Final save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(categories, f, indent=2)
    
    print(f"\nCategorization complete: {len(categories)}/{len(ingredients)} items")
    print(f"Saved to: {output_path}")
    
    # Statistics
    category_counts = {}
    for cat in categories.values():
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

def main():
    # Paths
    data_yaml = Path("data/merged/data.yaml")
    output_dir = Path("data/substitution")
    output_file = output_dir / "ingredient_categories.json"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ingredients
    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found")
        return
    
    ingredients = load_ingredient_names(data_yaml)
    print(f"Loaded {len(ingredients)} ingredients from {data_yaml}")
    
    # Load embedding model
    print("Loading sentence-transformers model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Interactive categorization
    interactive_categorization(ingredients, model, output_file)

if __name__ == "__main__":
    main()