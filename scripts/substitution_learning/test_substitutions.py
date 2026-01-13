"""
Substitution System Testing
Validates substitution quality through manual review with error taxonomy.
Compares against random baseline.
"""

import json
import random
from pathlib import Path
from substitution_ranker import SubstitutionRanker

# Test ingredients across categories (Western-focused, all mapped)
TEST_INGREDIENTS = {
    'proteins': ['Chicken', 'Beef', 'Pork', 'Egg', 'Tofu'],
    'vegetables': ['Tomato', 'Onion', 'Potato', 'Carrot', 'Broccoli', 'Cauliflower'],
    'spices': ['Garlic', 'Ginger', 'Chili Powder', 'Cinnamon', 'black pepper'],
    'grains': ['Rice -Chamal-', 'Wheat', 'Corn'],
    'dairy': ['Cheese', 'Milk', 'Butter']
}

ERROR_TYPES = {
    'good': 'Reasonable substitution',
    'ok': 'Acceptable but not ideal',
    'wrong_category': 'Different food category (protein->vegetable)',
    'wrong_cuisine': 'Incompatible cuisine context',
    'duplicate': 'Same ingredient, different name',
    'nonsensical': 'Completely unrelated'
}

def get_all_test_ingredients():
    """Flatten test ingredients into single list"""
    all_ingredients = []
    for category, ingredients in TEST_INGREDIENTS.items():
        all_ingredients.extend(ingredients)
    return all_ingredients

def random_baseline(ranker, ingredient, top_k=5):
    """Generate random substitution baseline"""
    available = [cls for cls in ranker.detection_classes if cls != ingredient]
    return random.sample(available, min(top_k, len(available)))

def display_substitution_for_review(ingredient, substitutions, ranker):
    """Display substitution and collect manual rating"""
    print(f"\n{'='*70}")
    print(f"Ingredient: {ingredient}")
    
    category = ranker.categories.get(ingredient, 'Unknown')
    print(f"Category: {category}")
    
    if not substitutions:
        print("No substitutions found")
        return []
    
    print(f"\nSubstitution candidates:")
    
    ratings = []
    for i, (candidate, score, pmi, confidence) in enumerate(substitutions, 1):
        candidate_category = ranker.categories.get(candidate, 'Unknown')
        
        print(f"\n  {i}. {candidate}")
        print(f"     Category: {candidate_category}")
        print(f"     Score: {score:.4f} | PMI: {pmi:.3f} | Confidence: {confidence}")
        
        # Get rating
        print(f"\n     Rate this substitution:")
        print(f"     1=good, 2=ok, 3=wrong_category, 4=wrong_cuisine, 5=duplicate, 6=nonsensical")
        
        while True:
            try:
                rating_input = input(f"     Rating (1-6) or 's' to skip: ").strip()
                
                if rating_input.lower() == 's':
                    rating = None
                    break
                
                rating_num = int(rating_input)
                if 1 <= rating_num <= 6:
                    rating_labels = ['good', 'ok', 'wrong_category', 'wrong_cuisine', 'duplicate', 'nonsensical']
                    rating = rating_labels[rating_num - 1]
                    break
                else:
                    print("     Invalid input. Enter 1-6 or 's'")
            except ValueError:
                print("     Invalid input. Enter 1-6 or 's'")
        
        if rating:
            ratings.append({
                'ingredient': ingredient,
                'candidate': candidate,
                'rank': i,
                'score': score,
                'pmi': pmi,
                'confidence': confidence,
                'rating': rating,
                'ingredient_category': category,
                'candidate_category': candidate_category
            })
    
    return ratings

def calculate_metrics(all_ratings):
    """Calculate precision and error distribution"""
    if not all_ratings:
        return {}
    
    # Precision @ K
    total_ratings = len(all_ratings)
    good_ratings = sum(1 for r in all_ratings if r['rating'] in ['good', 'ok'])
    
    precision = good_ratings / total_ratings if total_ratings > 0 else 0
    
    # Precision @ 1, 3, 5
    rank_1 = [r for r in all_ratings if r['rank'] == 1]
    rank_3 = [r for r in all_ratings if r['rank'] <= 3]
    rank_5 = [r for r in all_ratings if r['rank'] <= 5]
    
    p_at_1 = sum(1 for r in rank_1 if r['rating'] in ['good', 'ok']) / len(rank_1) if rank_1 else 0
    p_at_3 = sum(1 for r in rank_3 if r['rating'] in ['good', 'ok']) / len(rank_3) if rank_3 else 0
    p_at_5 = sum(1 for r in rank_5 if r['rating'] in ['good', 'ok']) / len(rank_5) if rank_5 else 0
    
    # Error distribution
    error_dist = {}
    for r in all_ratings:
        error_dist[r['rating']] = error_dist.get(r['rating'], 0) + 1
    
    # Confidence analysis
    conf_analysis = {'high': [], 'medium': [], 'low': []}
    for r in all_ratings:
        conf = r['confidence']
        is_good = r['rating'] in ['good', 'ok']
        conf_analysis[conf].append(is_good)
    
    conf_precision = {}
    for conf, results in conf_analysis.items():
        if results:
            conf_precision[conf] = sum(results) / len(results)
    
    return {
        'total_ratings': total_ratings,
        'good_ratings': good_ratings,
        'overall_precision': precision,
        'precision_at_1': p_at_1,
        'precision_at_3': p_at_3,
        'precision_at_5': p_at_5,
        'error_distribution': error_dist,
        'confidence_precision': conf_precision
    }

def save_results(all_ratings, metrics, output_path):
    """Save validation results"""
    results = {
        'ratings': all_ratings,
        'metrics': metrics,
        'test_ingredients': TEST_INGREDIENTS
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Also save human-readable report
    report_path = output_path.parent / "validation_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Substitution System Validation Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total ratings: {metrics['total_ratings']}\n")
        f.write(f"Good/OK ratings: {metrics['good_ratings']}\n")
        f.write(f"Overall precision: {metrics['overall_precision']:.2%}\n")
        f.write(f"Precision@1: {metrics['precision_at_1']:.2%}\n")
        f.write(f"Precision@3: {metrics['precision_at_3']:.2%}\n")
        f.write(f"Precision@5: {metrics['precision_at_5']:.2%}\n\n")
        
        f.write("ERROR DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        for error_type, count in sorted(metrics['error_distribution'].items(), key=lambda x: -x[1]):
            pct = count / metrics['total_ratings'] * 100
            f.write(f"{error_type}: {count} ({pct:.1f}%)\n")
        
        f.write("\nCONFIDENCE ANALYSIS\n")
        f.write("-" * 70 + "\n")
        for conf, precision in metrics['confidence_precision'].items():
            f.write(f"{conf}: {precision:.2%}\n")
        
        f.write("\nDETAILED RATINGS\n")
        f.write("-" * 70 + "\n")
        for rating in all_ratings:
            f.write(f"\n{rating['ingredient']} -> {rating['candidate']}\n")
            f.write(f"  Rank: {rating['rank']} | Score: {rating['score']:.4f} | PMI: {rating['pmi']:.3f}\n")
            f.write(f"  Rating: {rating['rating']} | Confidence: {rating['confidence']}\n")
    
    print(f"\nResults saved to:")
    print(f"  {output_path}")
    print(f"  {report_path}")

def main():
    # Paths
    data_yaml = Path("data/merged/data.yaml")
    embeddings_path = Path("data/embeddings/semantic_embeddings.pt")
    pmi_matrix_path = Path("data/substitution/pmi_matrix.npz")
    categories_path = Path("data/substitution/ingredient_categories.json")
    output_dir = Path("results/substitution")
    output_file = output_dir / "validation_results.json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check files
    required_files = [data_yaml, embeddings_path, pmi_matrix_path]
    for f in required_files:
        if not f.exists():
            print(f"Error: {f} not found")
            return
    
    # Initialize ranker
    print("Initializing substitution ranker...")
    ranker = SubstitutionRanker(
        data_yaml,
        embeddings_path,
        pmi_matrix_path,
        categories_path
    )
    
    print("\n" + "="*70)
    print("SUBSTITUTION SYSTEM VALIDATION")
    print("="*70)
    print("\nInstructions:")
    print("- Review each substitution candidate")
    print("- Rate using: 1=good, 2=ok, 3=wrong_category, 4=wrong_cuisine, 5=duplicate, 6=nonsensical")
    print("- Press 's' to skip a candidate")
    print("- Press Ctrl+C to exit and save progress")
    print("\nCategories being tested:")
    for category, ingredients in TEST_INGREDIENTS.items():
        print(f"  {category}: {', '.join(ingredients)}")
    
    input("\nPress Enter to begin...")
    
    all_ratings = []
    
    try:
        for category, ingredients in TEST_INGREDIENTS.items():
            print(f"\n\n{'#'*70}")
            print(f"CATEGORY: {category.upper()}")
            print('#'*70)
            
            for ingredient in ingredients:
                # Get substitutions
                substitutions = ranker.get_substitutions(ingredient, top_k=5)
                
                # Manual review
                ratings = display_substitution_for_review(ingredient, substitutions, ranker)
                all_ratings.extend(ratings)
    
    except KeyboardInterrupt:
        print("\n\nValidation interrupted. Saving progress...")
    
    # Calculate metrics
    print("\n\nCalculating metrics...")
    metrics = calculate_metrics(all_ratings)
    
    # Display summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Total ratings: {metrics['total_ratings']}")
    print(f"Good/OK ratings: {metrics['good_ratings']}")
    print(f"Overall precision: {metrics['overall_precision']:.2%}")
    print(f"\nPrecision by rank:")
    print(f"  P@1: {metrics['precision_at_1']:.2%}")
    print(f"  P@3: {metrics['precision_at_3']:.2%}")
    print(f"  P@5: {metrics['precision_at_5']:.2%}")
    print(f"\nError distribution:")
    for error_type, count in sorted(metrics['error_distribution'].items(), key=lambda x: -x[1]):
        pct = count / metrics['total_ratings'] * 100
        print(f"  {error_type}: {count} ({pct:.1f}%)")
    
    # Save results
    save_results(all_ratings, metrics, output_file)
    
    # Success criteria check
    print("\n" + "="*70)
    print("SUCCESS CRITERIA CHECK")
    print("="*70)
    
    min_precision = 0.60
    good_precision = 0.75
    
    if metrics['overall_precision'] >= good_precision:
        print(f"EXCELLENT: Precision {metrics['overall_precision']:.2%} >= {good_precision:.0%}")
    elif metrics['overall_precision'] >= min_precision:
        print(f"ACCEPTABLE: Precision {metrics['overall_precision']:.2%} >= {min_precision:.0%}")
    else:
        print(f"NEEDS IMPROVEMENT: Precision {metrics['overall_precision']:.2%} < {min_precision:.0%}")
        print("\nConsider:")
        print("  - Adjusting PMI thresholds")
        print("  - Using different embedding model")
        print("  - Adding manual substitution rules")

if __name__ == "__main__":
    main()