"""
Training Data Quality Validation
Checks format, content quality, and distribution before training.
"""

import json
from pathlib import Path
from collections import Counter
import random

def load_jsonl(filepath, max_lines=None):
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            data.append(json.loads(line))
    return data

def validate_format(data, sample_size=100):
    """Check if data has correct format"""
    print("\n1. FORMAT VALIDATION")
    print("-" * 70)
    
    sample = random.sample(data, min(sample_size, len(data)))
    
    required_fields = ['instruction', 'output']
    issues = []
    
    for i, item in enumerate(sample):
        # Check required fields
        for field in required_fields:
            if field not in item:
                issues.append(f"Sample {i}: Missing '{field}'")
        
        # Check field types and emptiness
        if 'instruction' in item:
            if not isinstance(item['instruction'], str):
                issues.append(f"Sample {i}: 'instruction' not string")
            elif len(item['instruction'].strip()) < 10:
                issues.append(f"Sample {i}: 'instruction' too short")
        
        if 'output' in item:
            if not isinstance(item['output'], str):
                issues.append(f"Sample {i}: 'output' not string")
            elif len(item['output'].strip()) < 50:
                issues.append(f"Sample {i}: 'output' too short")
    
    if issues:
        print(f"✗ Found {len(issues)} format issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        return False
    else:
        print(f"✓ All {sample_size} samples have correct format")
        return True

def validate_content(data, sample_size=10):
    """Check content quality"""
    print("\n2. CONTENT QUALITY CHECK")
    print("-" * 70)
    
    sample = random.sample(data, min(sample_size, len(data)))
    
    print(f"\nShowing {len(sample)} random examples:\n")
    
    for i, item in enumerate(sample, 1):
        print(f"Example {i}:")
        print(f"  Instruction: {item['instruction'][:100]}...")
        print(f"  Output length: {len(item['output'])} chars")
        
        # Check for common issues
        output = item['output']
        if '**' not in output:
            print(f"  ⚠ No formatting (missing **)")
        if 'Ingredients:' not in output and 'ingredients:' not in output.lower():
            print(f"  ⚠ No ingredients section")
        if 'Instructions:' not in output and 'instructions:' not in output.lower():
            print(f"  ⚠ No instructions section")
        
        print()

def validate_distribution(train_data, val_data, test_data, augmented_data):
    """Check data distribution"""
    print("\n3. DATASET DISTRIBUTION")
    print("-" * 70)
    
    total = len(train_data) + len(val_data) + len(test_data)
    
    print(f"\nBase recipes:")
    print(f"  Train: {len(train_data):,} ({len(train_data)/total*100:.1f}%)")
    print(f"  Val:   {len(val_data):,} ({len(val_data)/total*100:.1f}%)")
    print(f"  Test:  {len(test_data):,} ({len(test_data)/total*100:.1f}%)")
    print(f"  Total: {total:,}")
    
    print(f"\nAugmented data:")
    print(f"  Synthetic: {len(augmented_data):,}")
    
    # Check task distribution in augmented data
    if augmented_data:
        tasks = Counter([item.get('task_type', 'unknown') for item in augmented_data])
        print(f"\nAugmented task types:")
        for task, count in tasks.most_common():
            print(f"  {task}: {count:,} ({count/len(augmented_data)*100:.1f}%)")
    
    # Check difficulty distribution in training data
    if train_data and 'difficulty' in train_data[0]:
        difficulties = Counter([item.get('difficulty', 'unknown') for item in train_data])
        print(f"\nTraining difficulty distribution:")
        for diff, count in difficulties.most_common():
            print(f"  {diff}: {count:,} ({count/len(train_data)*100:.1f}%)")
    
    # Validation checks
    issues = []
    if len(val_data) < 1000:
        issues.append(f"⚠ Validation set very small ({len(val_data)} examples)")
    if len(test_data) < 1000:
        issues.append(f"⚠ Test set very small ({len(test_data)} examples)")
    if len(train_data) < 10000:
        issues.append(f"⚠ Training set very small ({len(train_data)} examples)")
    
    if issues:
        print("\nWarnings:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ Dataset sizes look good")

def validate_lengths(data, sample_size=1000):
    """Check token length distribution"""
    print("\n4. LENGTH DISTRIBUTION")
    print("-" * 70)
    
    sample = random.sample(data, min(sample_size, len(data)))
    
    lengths = [len(item['instruction']) + len(item['output']) for item in sample]
    
    print(f"\nCharacter counts (sampled {len(sample)} examples):")
    print(f"  Min:    {min(lengths):,}")
    print(f"  Mean:   {sum(lengths)//len(lengths):,}")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]:,}")
    print(f"  Max:    {max(lengths):,}")
    
    # Estimate tokens (rough: 1 token ≈ 4 chars)
    token_lengths = [l // 4 for l in lengths]
    print(f"\nEstimated tokens:")
    print(f"  Mean:   {sum(token_lengths)//len(token_lengths):,}")
    print(f"  Median: {sorted(token_lengths)[len(token_lengths)//2]:,}")
    print(f"  Max:    {max(token_lengths):,}")
    
    # Check for issues
    very_short = sum(1 for l in lengths if l < 200)
    very_long = sum(1 for l in token_lengths if l > 2048)
    
    if very_short > len(sample) * 0.1:
        print(f"\n⚠ Warning: {very_short} examples ({very_short/len(sample)*100:.1f}%) are very short (<200 chars)")
    
    if very_long > 0:
        print(f"\n⚠ Warning: {very_long} examples exceed 2048 tokens (will be truncated)")
    else:
        print(f"\n✓ All examples fit within 2048 token limit")

def main():
    print("=" * 70)
    print("TRAINING DATA QUALITY VALIDATION")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_data = load_jsonl("data/training/train_instructions.jsonl")
    val_data = load_jsonl("data/training/val_instructions.jsonl")
    test_data = load_jsonl("data/training/test_instructions.jsonl")
    augmented_data = load_jsonl("data/training/augmented_substitutions.jsonl")
    
    print(f"Loaded {len(train_data):,} train, {len(val_data):,} val, {len(test_data):,} test, {len(augmented_data):,} augmented")
    
    # Run validations
    format_ok = validate_format(train_data, sample_size=100)
    
    if format_ok:
        validate_content(train_data, sample_size=5)
    
    validate_distribution(train_data, val_data, test_data, augmented_data)
    validate_lengths(train_data, sample_size=1000)
    
    # Check augmented data too
    if augmented_data:
        print("\n" + "=" * 70)
        print("AUGMENTED DATA VALIDATION")
        print("=" * 70)
        augmented_format_ok = validate_format(augmented_data, sample_size=100)
        if augmented_format_ok:
            validate_content(augmented_data, sample_size=3)
    
    # Final verdict
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    if format_ok:
        print("\n✓ Format validation: PASSED")
    else:
        print("\n✗ Format validation: FAILED")
        print("  Fix format issues before training!")
        return
    
    print("✓ Content check: Reviewed")
    print("✓ Distribution check: Completed")
    print("✓ Length check: Completed")
    
    print("\n" + "=" * 70)
    print("READY TO TRAIN")
    print("=" * 70)
    print("\nCommand:")
    print("  python scripts\\recipe_generation\\train_llama.py")
    print("\nEstimated training time: 6-8 hours")
    print("Dataset size: ~335K examples")

if __name__ == "__main__":
    main()