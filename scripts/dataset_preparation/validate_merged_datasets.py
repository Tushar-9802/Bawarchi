#!/usr/bin/env python3
"""
BAWARCHI - Merged Dataset Validation
Verifies integrity, class distribution, and annotation quality
"""

from pathlib import Path
import yaml
from collections import defaultdict

MERGED_DIR = Path("C:/Users/Tushar/Desktop/bawarchi/data/merged")

print("=" * 70)
print("MERGED DATASET VALIDATION")
print("=" * 70)


def validate_merged_dataset():
    """Validate merged dataset structure and content"""
    
    # Check data.yaml exists
    data_yaml_path = MERGED_DIR / "data.yaml"
    if not data_yaml_path.exists():
        print("✗ data.yaml not found!")
        return False
    
    # Load data.yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    nc = data['nc']
    names = data['names']
    
    print(f"\n[DATASET CONFIG]")
    print(f"✓ data.yaml loaded")
    print(f"  Classes (nc): {nc}")
    print(f"  Class names: {len(names)}")
    
    if nc != len(names):
        print(f"✗ ERROR: nc ({nc}) != len(names) ({len(names)})")
        return False
    
    # Validate splits
    print(f"\n[SPLIT VALIDATION]")
    splits = ['train', 'valid', 'test']
    split_stats = {}
    
    for split in splits:
        img_dir = MERGED_DIR / split / 'images'
        lbl_dir = MERGED_DIR / split / 'labels'
        
        if not img_dir.exists():
            print(f"✗ {split}/images directory not found")
            continue
        
        if not lbl_dir.exists():
            print(f"✗ {split}/labels directory not found")
            continue
        
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt"))
        
        # Check image-label pairs
        matched = 0
        for img in images:
            label_file = lbl_dir / f"{img.stem}.txt"
            if label_file.exists():
                matched += 1
        
        split_stats[split] = {
            'images': len(images),
            'labels': len(labels),
            'matched': matched,
            'missing_labels': len(images) - matched
        }
        
        print(f"\n{split}:")
        print(f"  Images: {len(images):,}")
        print(f"  Labels: {len(labels):,}")
        print(f"  Matched pairs: {matched:,}")
        if len(images) - matched > 0:
            print(f"  ⚠ Missing labels: {len(images) - matched}")
    
    # Class distribution analysis
    print(f"\n[CLASS DISTRIBUTION]")
    class_counts = defaultdict(int)
    total_annotations = 0
    
    for split in splits:
        lbl_dir = MERGED_DIR / split / 'labels'
        if not lbl_dir.exists():
            continue
        
        for label_file in lbl_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if 0 <= class_id < nc:
                            class_counts[class_id] += 1
                            total_annotations += 1
                        else:
                            print(f"⚠ Invalid class ID {class_id} in {label_file.name}")
    
    print(f"\nTotal annotations: {total_annotations:,}")
    print(f"Classes with data: {len(class_counts)}/{nc}")
    
    # Show top 10 and bottom 10 classes
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n[TOP 10 CLASSES]")
    for i, (class_id, count) in enumerate(sorted_classes[:10], 1):
        class_name = names[class_id] if class_id < len(names) else "Unknown"
        print(f"  {i}. {class_name}: {count:,} annotations")
    
    print(f"\n[BOTTOM 10 CLASSES]")
    for i, (class_id, count) in enumerate(sorted_classes[-10:], 1):
        class_name = names[class_id] if class_id < len(names) else "Unknown"
        print(f"  {i}. {class_name}: {count} annotations")
    
    # Classes with zero annotations
    missing_classes = set(range(nc)) - set(class_counts.keys())
    if missing_classes:
        print(f"\n[CLASSES WITH NO ANNOTATIONS]")
        print(f"⚠ {len(missing_classes)} classes have no training data:")
        for class_id in sorted(missing_classes)[:20]:  # Show first 20
            class_name = names[class_id] if class_id < len(names) else "Unknown"
            print(f"  Class {class_id}: {class_name}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    total_images = sum(s['images'] for s in split_stats.values())
    total_matched = sum(s['matched'] for s in split_stats.values())
    
    print(f"\n✓ Dataset location: {MERGED_DIR}")
    print(f"✓ Total images: {total_images:,}")
    print(f"✓ Total annotations: {total_annotations:,}")
    print(f"✓ Classes: {nc}")
    print(f"✓ Classes with data: {len(class_counts)}/{nc}")
    
    if missing_classes:
        print(f"⚠ Classes without data: {len(missing_classes)}")
    
    if total_images == total_matched:
        print(f"✓ All images have labels")
    else:
        print(f"⚠ {total_images - total_matched} images missing labels")
    
    # Readiness check
    print(f"\n[TRAINING READINESS]")
    if total_images >= 1000 and len(class_counts) >= 10:
        print("✓ READY FOR TRAINING")
        print(f"  Sufficient images: {total_images:,} ≥ 1,000")
        print(f"  Sufficient classes: {len(class_counts)} ≥ 10")
    else:
        print("⚠ NOT READY - Need more data")
        if total_images < 1000:
            print(f"  Need {1000 - total_images:,} more images")
        if len(class_counts) < 10:
            print(f"  Need {10 - len(class_counts)} more classes with data")
    
    print("\n" + "=" * 70)
    
    return True


if __name__ == "__main__":
    validate_merged_dataset()