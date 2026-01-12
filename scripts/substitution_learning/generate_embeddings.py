"""
Semantic Embedding Generator
Generates enhanced embeddings for ingredient names using all-mpnet-base-v2.
Includes context expansion with category information.
"""

import yaml
import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

def load_detection_classes(data_yaml_path):
    """Load detection class names"""
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['names']

def load_categories(categories_path):
    """Load ingredient categories"""
    if not categories_path.exists():
        print(f"Warning: {categories_path} not found. Embeddings will be generated without category context.")
        return {}
    
    with open(categories_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def expand_context(ingredient_name, category=None):
    """
    Expand ingredient name with contextual information
    Example: "Paneer" → "paneer cheese protein dairy indian cottage"
    """
    # Clean name
    name_clean = ingredient_name.lower()
    name_clean = name_clean.replace('-', ' ')
    
    # Start with clean name
    context = [name_clean]
    
    # Add category if available
    if category:
        context.append(category)
        
        # Add category-specific keywords
        category_keywords = {
            'protein': ['protein', 'meat', 'source'],
            'vegetable': ['vegetable', 'produce', 'fresh'],
            'spice': ['spice', 'seasoning', 'flavor'],
            'grain': ['grain', 'staple', 'carbohydrate'],
            'dairy': ['dairy', 'milk', 'cream'],
            'fruit': ['fruit', 'sweet', 'fresh'],
            'condiment': ['condiment', 'sauce', 'flavor'],
            'fat': ['fat', 'oil', 'cooking']
        }
        
        if category in category_keywords:
            context.extend(category_keywords[category])
    
    # Detect cuisine markers in name
    if any(word in name_clean for word in ['paneer', 'masala', 'dal', 'roti', 'ghee']):
        context.append('indian')
    if any(word in name_clean for word in ['tortilla', 'salsa', 'taco', 'chile']):
        context.append('mexican')
    
    return ' '.join(context)

def generate_embeddings(detection_classes, categories, model):
    """Generate embeddings for all detection classes"""
    print(f"Generating embeddings for {len(detection_classes)} ingredients...")
    
    # Prepare contexts
    contexts = []
    for ingredient in detection_classes:
        category = categories.get(ingredient)
        context = expand_context(ingredient, category)
        contexts.append(context)
    
    # Generate embeddings
    embeddings = model.encode(contexts, show_progress_bar=True, convert_to_tensor=True)
    
    # Convert to CPU tensor for saving
    embeddings = embeddings.cpu()
    
    return embeddings

def main():
    # Paths
    data_yaml = Path("data/merged/data.yaml")
    categories_file = Path("data/substitution/ingredient_categories.json")
    output_dir = Path("data/embeddings")
    output_file = output_dir / "semantic_embeddings.pt"
    names_file = output_dir / "ingredient_names.txt"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load detection classes
    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found")
        return
    
    detection_classes = load_detection_classes(data_yaml)
    print(f"Loaded {len(detection_classes)} detection classes")
    
    # Load categories (optional)
    categories = load_categories(categories_file)
    if categories:
        print(f"Loaded categories for {len(categories)} ingredients")
    
    # Load model
    print("\nLoading sentence-transformers model: all-mpnet-base-v2")
    print("This model produces 768-dimensional embeddings")
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # Generate embeddings
    embeddings = generate_embeddings(detection_classes, categories, model)
    
    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    
    # Save embeddings
    torch.save(embeddings, output_file)
    print(f"Saved embeddings to {output_file}")
    
    # Save ingredient names (for reference)
    with open(names_file, 'w', encoding='utf-8') as f:
        for name in detection_classes:
            f.write(f"{name}\n")
    print(f"Saved ingredient names to {names_file}")
    
    # Sample similarity check
    print("\nSample similarity check (first 5 ingredients):")
    for i in range(min(5, len(detection_classes))):
        # Find most similar ingredient
        similarities = torch.nn.functional.cosine_similarity(
            embeddings[i].unsqueeze(0),
            embeddings,
            dim=1
        )
        
        # Exclude self
        similarities[i] = -1
        
        # Get top 3
        top_3_idx = torch.topk(similarities, 3).indices
        
        print(f"\n  {detection_classes[i]}:")
        for idx in top_3_idx:
            sim_score = similarities[idx].item()
            print(f"    - {detection_classes[idx]}: {sim_score:.3f}")
    
    # Save metadata
    metadata = {
        'model': 'all-mpnet-base-v2',
        'dimensions': embeddings.shape[1],
        'num_ingredients': embeddings.shape[0],
        'context_expansion': 'enabled' if categories else 'disabled'
    }
    
    with open(output_dir / "embeddings_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nEmbedding generation complete")

if __name__ == "__main__":
    main()