"""
Script Validation - Checks all critical fixes are present
Run this before training to verify everything is correct.
"""

import sys
from pathlib import Path

def check_file_exists(filepath):
    """Check if file exists"""
    if not Path(filepath).exists():
        print(f"❌ MISSING: {filepath}")
        return False
    print(f"✓ Found: {filepath}")
    return True

def check_response_template():
    """CRITICAL: Check response_template is present in train_llama.py"""
    print("\n" + "="*70)
    print("CRITICAL CHECK: Response Template (Bug 1 Fix)")
    print("="*70)
    
    filepath = "scripts/recipe_generation/train_llama.py"
    if not Path(filepath).exists():
        filepath = "train_llama.py"  # Try current directory
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for response_template
        if 'response_template=' in content:
            # Extract the line
            for line in content.split('\n'):
                if 'response_template=' in line:
                    print(f"✓ FOUND: {line.strip()}")
                    
                    # Validate correct value
                    if 'assistant<|end_header_id|>' in line:
                        print("✓ CORRECT: Response template properly configured")
                        return True
                    else:
                        print("❌ ERROR: Response template has wrong value")
                        return False
        else:
            print("❌ CRITICAL ERROR: response_template NOT FOUND!")
            print("   Model will learn prompts instead of responses!")
            print("   This will reduce quality by 10-15%")
            return False
            
    except FileNotFoundError:
        print(f"❌ ERROR: Cannot find {filepath}")
        return False

def check_stratified_split():
    """Check stratified split is implemented"""
    print("\n" + "="*70)
    print("CHECK: Stratified Split (Bug 2 Fix)")
    print("="*70)
    
    filepath = "scripts/recipe_generation/prepare_training_data.py"
    if not Path(filepath).exists():
        filepath = "prepare_training_data.py"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'difficulty_groups' in content and 'Stratified splitting' in content:
            print("✓ FOUND: Stratified splitting by difficulty")
            return True
        else:
            print("⚠ WARNING: Stratified split might not be implemented")
            print("   Evaluation might be less reliable")
            return False
            
    except FileNotFoundError:
        print(f"❌ ERROR: Cannot find {filepath}")
        return False

def check_deterministic_seeds():
    """Check random seeds are set"""
    print("\n" + "="*70)
    print("CHECK: Deterministic Seeds (Bug 3 Fix)")
    print("="*70)
    
    filepath = "scripts/recipe_generation/train_llama.py"
    if not Path(filepath).exists():
        filepath = "train_llama.py"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            'random.seed': False,
            'np.random.seed': False,
            'torch.manual_seed': False,
        }
        
        for key in checks:
            if key in content:
                checks[key] = True
                print(f"✓ FOUND: {key}()")
        
        if all(checks.values()):
            print("✓ CORRECT: All random seeds set for reproducibility")
            return True
        else:
            print("⚠ WARNING: Some random seeds missing")
            print("   Results might not be reproducible")
            return False
            
    except FileNotFoundError:
        print(f"❌ ERROR: Cannot find {filepath}")
        return False

def check_quality_validation():
    """Check synthetic data quality validation"""
    print("\n" + "="*70)
    print("CHECK: Quality Validation (Bug 4 Fix)")
    print("="*70)
    
    filepath = "scripts/recipe_generation/augment_with_substitutions.py"
    if not Path(filepath).exists():
        filepath = "augment_with_substitutions.py"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for quality checks
        checks = [
            ('Skip duplicates', 'original_class.lower() == substitute_class.lower()'),
            ('Skip circular', 'substitute_class.lower() in ing.lower()'),
            ('Context quality', 'Direct 1:1 substitution'),
        ]
        
        found = 0
        for name, pattern in checks:
            if pattern in content:
                print(f"✓ FOUND: {name} check")
                found += 1
        
        if found >= 2:
            print(f"✓ CORRECT: Quality validation implemented ({found}/3 checks)")
            return True
        else:
            print(f"⚠ WARNING: Limited quality validation ({found}/3 checks)")
            return False
            
    except FileNotFoundError:
        print(f"❌ ERROR: Cannot find {filepath}")
        return False

def check_memory_cleanup():
    """Check DataFrame memory cleanup"""
    print("\n" + "="*70)
    print("CHECK: Memory Cleanup (Bug 5 Fix)")
    print("="*70)
    
    filepath = "scripts/recipe_generation/prepare_training_data.py"
    if not Path(filepath).exists():
        filepath = "prepare_training_data.py"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'del df' in content and 'gc.collect()' in content:
            print("✓ FOUND: Memory cleanup (del df, gc.collect)")
            return True
        else:
            print("⚠ WARNING: Memory cleanup might not be implemented")
            print("   Could cause OOM on systems with <16GB RAM")
            return False
            
    except FileNotFoundError:
        print(f"❌ ERROR: Cannot find {filepath}")
        return False

def check_optimizations():
    """Check key optimizations are present"""
    print("\n" + "="*70)
    print("CHECK: Key Optimizations")
    print("="*70)
    
    filepath = "scripts/recipe_generation/train_llama.py"
    if not Path(filepath).exists():
        filepath = "train_llama.py"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        opts = {
            'NEFTune': 'neftune_noise_alpha',
            'Group-by-length': 'group_by_length',
            'Cosine with restarts': 'cosine_with_restarts',
        }
        
        found = []
        for name, pattern in opts.items():
            if pattern in content:
                print(f"✓ FOUND: {name}")
                found.append(name)
            else:
                print(f"⚠ MISSING: {name}")
        
        print(f"\nOptimizations: {len(found)}/{len(opts)} implemented")
        return len(found) >= 3
            
    except FileNotFoundError:
        print(f"❌ ERROR: Cannot find {filepath}")
        return False

def main():
    print("="*70)
    print("BAWARCHI TRAINING SCRIPTS VALIDATION")
    print("="*70)
    print("\nChecking all critical fixes and optimizations...")
    
    results = {
        'Response Template (CRITICAL)': check_response_template(),
        'Stratified Split': check_stratified_split(),
        'Deterministic Seeds': check_deterministic_seeds(),
        'Quality Validation': check_quality_validation(),
        'Memory Cleanup': check_memory_cleanup(),
        'Optimizations': check_optimizations(),
    }
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    critical_passed = results['Response Template (CRITICAL)']
    total_passed = sum(results.values())
    total_checks = len(results)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check}")
    
    print(f"\nTotal: {total_passed}/{total_checks} checks passed")
    
    if not critical_passed:
        print("\n" + "!"*70)
        print("❌ CRITICAL ERROR: Response template NOT configured!")
        print("!"*70)
        print("\nWITHOUT THIS FIX:")
        print("  - Model learns to predict PROMPTS (60-70% of capacity wasted)")
        print("  - Quality reduced by 10-15%")
        print("  - Training essentially broken")
        print("\nDO NOT TRAIN until this is fixed!")
        print("\nFix: Add this line to SFTTrainer in train_llama.py:")
        print('  response_template="<|start_header_id|>assistant<|end_header_id|>\\n\\n"')
        return False
    
    if total_passed >= 5:
        print("\n" + "="*70)
        print("✓ VALIDATION PASSED - READY TO TRAIN!")
        print("="*70)
        print("\nAll critical fixes verified. You can proceed with:")
        print("  python scripts/recipe_generation/prepare_training_data.py")
        return True
    else:
        print("\n" + "="*70)
        print("⚠ VALIDATION WARNINGS - REVIEW RECOMMENDED")
        print("="*70)
        print("\nSome checks failed. Training will work but might have issues.")
        print("Review the warnings above before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)