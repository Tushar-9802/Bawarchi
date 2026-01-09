#!/usr/bin/env python3
"""
Extract and process Recipe NLG dataset
Converts CSV to clean JSON format for T5 training
"""

import csv
import json
import re
from pathlib import Path
from collections import defaultdict


def clean_text(text):
    """Clean and normalize text"""
    if not text or text == 'nan' or str(text).lower() == 'nan':
        return ""
    
    text = str(text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s,.\-():/]', '', text)
    
    return text.strip()


def parse_ingredients(ingredient_string):
    """Parse ingredient string into list"""
    if not ingredient_string or str(ingredient_string).lower() == 'nan':
        return []
    
    ingredient_string = str(ingredient_string)
    
    # Try different separators
    if ',' in ingredient_string:
        ingredients = ingredient_string.split(',')
    elif ';' in ingredient_string:
        ingredients = ingredient_string.split(';')
    elif '|' in ingredient_string:
        ingredients = ingredient_string.split('|')
    else:
        # Single ingredient or newline-separated
        ingredients = ingredient_string.split('\n')
    
    # Clean each ingredient
    cleaned = []
    for ing in ingredients:
        ing = clean_text(ing)
        if ing and len(ing) > 1:  # Skip single characters
            cleaned.append(ing.lower())
    
    return cleaned


def parse_directions(direction_string):
    """Parse directions into structured steps"""
    if not direction_string or str(direction_string).lower() == 'nan':
        return []
    
    direction_string = str(direction_string)
    
    # Split by common delimiters
    steps = re.split(r'(?:\d+\.|Step \d+:?|\n\n)', direction_string)
    
    # Clean and filter
    cleaned_steps = []
    for step in steps:
        step = clean_text(step)
        if step and len(step) > 10:  # Skip very short steps
            cleaned_steps.append(step)
    
    return cleaned_steps


def extract_recipe_nlg(input_path, output_dir):
    """Extract Recipe NLG CSV to structured JSON"""
    
    print("=" * 60)
    print("Recipe NLG Extraction")
    print("=" * 60)
    print()
    
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"✗ File not found: {input_path}")
        return False
    
    # Check file size
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"File: {input_path}")
    print(f"Size: {file_size_mb:.1f} MB")
    
    if file_size_mb < 100:
        print(f"\n⚠️  WARNING: File is only {file_size_mb:.1f} MB")
        print("Expected size: 400-2000 MB for full Recipe NLG dataset")
        print()
    
    print(f"Output directory: {output_dir}")
    print()
    
    recipes = []
    skipped = 0
    ingredient_counts = defaultdict(int)
    
    print("Processing recipes...")
    print("This will take 5-10 minutes for 2M+ recipes...")
    print()
    
    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Try to detect delimiter
            sample = f.read(8192)
            f.seek(0)
            
            # Check what delimiter is used
            if '\t' in sample and sample.count('\t') > sample.count(','):
                delimiter = '\t'
                print("Detected: Tab-delimited file")
            else:
                delimiter = ','
                print("Detected: Comma-delimited file")
            
            reader = csv.DictReader(f, delimiter=delimiter)
            
            # Verify header
            if not reader.fieldnames:
                print("✗ No header row found")
                return False
            
            print(f"Columns: {', '.join(reader.fieldnames)}")
            print()
            
            for i, row in enumerate(reader, 1):
                # Progress updates
                if i % 50000 == 0:
                    print(f"  Processed {i:,} recipes... ({len(recipes):,} valid)")
                
                # Extract fields (handle different column names)
                title = clean_text(row.get('title') or row.get('name') or row.get('Title') or '')
                
                # Try different ingredient column names
                ingredients_raw = (row.get('ingredients') or row.get('NER') or 
                                  row.get('Ingredients') or '')
                ingredients = parse_ingredients(ingredients_raw)
                
                # Try different direction column names
                directions_raw = (row.get('directions') or row.get('instructions') or 
                                 row.get('steps') or row.get('Directions') or '')
                directions = parse_directions(directions_raw)
                
                # Skip invalid recipes
                if not title or len(ingredients) < 3 or len(directions) < 1:
                    skipped += 1
                    continue
                
                # Skip recipes with too many ingredients (likely garbage)
                if len(ingredients) > 30:
                    skipped += 1
                    continue
                
                # Build recipe object
                recipe = {
                    'id': f'recipe_nlg_{i}',
                    'title': title,
                    'ingredients': ingredients,
                    'instructions': directions,
                    'ingredient_count': len(ingredients),
                    'step_count': len(directions)
                }
                
                recipes.append(recipe)
                
                # Track ingredient frequency
                for ing in ingredients:
                    ingredient_counts[ing] += 1
        
        print(f"\n✓ Processed {i:,} total rows")
        print(f"✓ Extracted {len(recipes):,} valid recipes")
        print(f"✗ Skipped {skipped:,} invalid recipes")
        
        # Check if we got reasonable results
        if len(recipes) == 0:
            print("\n✗ No valid recipes extracted!")
            print("\nPossible issues:")
            print("1. CSV file is corrupted or incomplete")
            print("2. Wrong delimiter detected")
            print("3. Column names don't match expected format")
            return False
        
        if len(recipes) < 10000:
            print(f"\n⚠️  Only {len(recipes):,} recipes extracted")
            print("Expected: 100,000+ for partial dataset, 2,000,000+ for full dataset")
            print()
            response = input("Continue with limited data? (y/n): ")
            if response.lower() != 'y':
                return False
        
    except Exception as e:
        print(f"\n✗ Error reading CSV: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    avg_ingredients = sum(r['ingredient_count'] for r in recipes) / len(recipes)
    avg_steps = sum(r['step_count'] for r in recipes) / len(recipes)
    
    print(f"Total recipes: {len(recipes):,}")
    print(f"Unique ingredients: {len(ingredient_counts):,}")
    print(f"Avg ingredients per recipe: {avg_ingredients:.1f}")
    print(f"Avg steps per recipe: {avg_steps:.1f}")
    
    # Top ingredients
    print("\nTop 20 ingredients:")
    top_ingredients = sorted(ingredient_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    for ing, count in top_ingredients:
        print(f"  {ing}: {count:,} recipes")
    
    # Save processed recipes
    print("\n" + "=" * 60)
    print("Saving processed data...")
    print("=" * 60)
    
    # Save full dataset (may be very large)
    print("\n⚠️  Skipping full recipes_processed.json (too large)")
    print("Saving only splits and summaries...")
    
    # Save ingredient vocabulary
    vocab_file = output_dir / 'ingredient_vocab.json'
    vocab = {
        'ingredients': list(ingredient_counts.keys()),
        'count': len(ingredient_counts),
        'frequencies': dict(sorted(ingredient_counts.items(), key=lambda x: x[1], reverse=True)[:1000])
    }
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved: {vocab_file} (ingredient vocabulary)")
    
    # Split into train/val/test
    print("\nSplitting dataset...")
    
    from random import shuffle, seed
    seed(42)  # Reproducible splits
    
    shuffled = recipes.copy()
    shuffle(shuffled)
    
    train_size = int(len(shuffled) * 0.8)
    val_size = int(len(shuffled) * 0.1)
    
    train_recipes = shuffled[:train_size]
    val_recipes = shuffled[train_size:train_size + val_size]
    test_recipes = shuffled[train_size + val_size:]
    
    # Save splits
    splits = {
        'train': train_recipes,
        'val': val_recipes,
        'test': test_recipes
    }
    
    for split_name, split_data in splits.items():
        split_file = output_dir / f'{split_name}.json'
        
        print(f"Saving {split_name} split ({len(split_data):,} recipes)...")
        
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False)
        
        print(f"✓ Saved: {split_file}")
    
    # Create summary (small file for git)
    summary = {
        'total_recipes': len(recipes),
        'unique_ingredients': len(ingredient_counts),
        'avg_ingredients_per_recipe': round(avg_ingredients, 2),
        'avg_steps_per_recipe': round(avg_steps, 2),
        'splits': {
            'train': len(train_recipes),
            'val': len(val_recipes),
            'test': len(test_recipes)
        },
        'top_20_ingredients': [ing for ing, _ in top_ingredients],
        'file_locations': {
            'train': 'data/recipes/processed/train.json',
            'val': 'data/recipes/processed/val.json',
            'test': 'data/recipes/processed/test.json',
            'vocab': 'data/recipes/processed/ingredient_vocab.json'
        }
    }
    
    summary_file = output_dir / 'dataset_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved: {summary_file}")
    
    print("\n" + "=" * 60)
    print("✓ Extraction Complete!")
    print("=" * 60)
    print(f"\nOutput location: {output_dir.absolute()}")
    print("\nFiles created:")
    print(f"  - train.json ({len(train_recipes):,} recipes)")
    print(f"  - val.json ({len(val_recipes):,} recipes)")
    print(f"  - test.json ({len(test_recipes):,} recipes)")
    print(f"  - ingredient_vocab.json (vocabulary)")
    print(f"  - dataset_summary.json (statistics)")
    
    return True


def main():
    """Main extraction function"""
    
    input_file = Path("data/recipes/raw/recipe_nlg.csv")
    output_dir = Path("data/recipes/processed")
    
    if not input_file.exists():
        print("=" * 60)
        print("ERROR: Recipe NLG CSV not found")
        print("=" * 60)
        print()
        print(f"Expected location: {input_file.absolute()}")
        print()
        print("Please download first:")
        print("  python scripts/download_recipes.py")
        print()
        return False
    
    success = extract_recipe_nlg(input_file, output_dir)
    
    if success:
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("1. Review: data/recipes/processed/dataset_summary.json")
        print("2. Week 4: Build substitution taxonomy")
        print("3. Week 5: Train T5 model")
    else:
        print("\n" + "=" * 60)
        print("EXTRACTION FAILED")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Check CSV file size (should be 400+ MB)")
        print("2. Check CSV format (open in text editor)")
        print("3. Try re-downloading Recipe NLG")
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)