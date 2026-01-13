"""
Fix Ingredient Categories
Corrects mis-categorizations identified during validation.
"""

import json
from pathlib import Path

def fix_categories():
    """Apply corrections to ingredient categories"""
    
    categories_file = Path("data/substitution/ingredient_categories.json")
    
    if not categories_file.exists():
        print(f"Error: {categories_file} not found")
        return
    
    # Load current categories
    with open(categories_file, 'r', encoding='utf-8') as f:
        categories = json.load(f)
    
    print(f"Loaded {len(categories)} ingredient categories")
    
    # Corrections identified from validation
    corrections = {
        # Aromatics/spices mis-categorized as vegetables
        'Garlic': 'spice',  # Was: vegetable
        
        # Vegetables mis-categorized as grains
        'Garden Peas': 'vegetable',  # Was: grain
        'Chayote-iskus-': 'vegetable',  # Was: grain
        'Farsi ko Munta': 'vegetable',  # Was: grain
        'Garden cress-Chamsur ko saag-': 'vegetable',  # Was: grain
        
        # Lentils mis-categorized as grains (should be protein)
        'Black Lentils': 'protein',  # Was: grain
        'Green Lentils': 'protein',  # Was: grain
        'Red Lentils': 'protein',  # Was: grain
        'Yellow Lentils': 'protein',  # Was: grain
        'Rahar ko Daal': 'protein',  # Was: grain
        
        # Critical errors - non-dairy as dairy
        'Gundruk': 'condiment',  # Was: dairy (fermented greens)
        'Cassava -Ghar Tarul-': 'vegetable',  # Was: dairy (root vegetable)
        
        # Grains mis-categorized
        'noodle': 'grain',  # Was: protein
        'Cornflakec': 'grain',  # Verify this is correct
        'Thukpa Noodles': 'grain',  # Was: varies
        
        # Other corrections
        'Ice': 'condiment',  # Was: varies (water/ice)
        'Pea': 'vegetable',  # Was: varies
        
        # Additional aromatic corrections
        'Ginger': 'spice',  # Verify current
        'Onion Leaves': 'spice',  # Was: vegetable (used as aromatic)
        'Coriander -Dhaniya-': 'spice',  # Was: varies
        'Green Mint -Pudina-': 'spice',  # Was: varies
    }
    
    # Apply corrections
    changes_made = []
    for ingredient, new_category in corrections.items():
        if ingredient in categories:
            old_category = categories[ingredient]
            if old_category != new_category:
                categories[ingredient] = new_category
                changes_made.append((ingredient, old_category, new_category))
                print(f"  {ingredient}: {old_category} → {new_category}")
        else:
            print(f"  Warning: {ingredient} not found in categories")
    
    print(f"\nApplied {len(changes_made)} corrections")
    
    # Save backup
    backup_file = categories_file.parent / "ingredient_categories_backup.json"
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(categories, f, indent=2)
    print(f"Backup saved to {backup_file}")
    
    # Save updated categories
    with open(categories_file, 'w', encoding='utf-8') as f:
        json.dump(categories, f, indent=2)
    print(f"Updated categories saved to {categories_file}")
    
    # Statistics
    category_counts = {}
    for cat in categories.values():
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nUpdated category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    # List of changes for report
    print("\nChanges made:")
    for ingredient, old_cat, new_cat in changes_made:
        print(f"  - {ingredient}: {old_cat} → {new_cat}")

def main():
    print("Fixing ingredient categories...")
    print("=" * 70)
    fix_categories()
    print("=" * 70)
    print("\nCategory fixes complete!")
    print("\nNext steps:")
    print("1. Run: python scripts\\substitution_learning\\add_grain_rules.py")
    print("2. Test: python scripts\\substitution_learning\\substitution_ranker.py")
    print("3. Validate: python scripts\\substitution_learning\\test_substitutions.py")

if __name__ == "__main__":
    main()