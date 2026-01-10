"""
ROBOFLOW DATASET VALIDATOR
Verifies YOLO format dataset integrity, class distribution, annotation quality.
"""

import os
import yaml
from pathlib import Path
from collections import Counter
import cv2

def validate_roboflow_dataset(dataset_root: str):
    """
    Validate Roboflow YOLO dataset structure and quality.
    
    Expected structure:
    dataset_root/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/ (optional)
    │   ├── images/
    │   └── labels/
    └── data.yaml
    """
    
    root = Path(dataset_root)
    print("="*60)
    print("ROBOFLOW DATASET VALIDATION")
    print("="*60)
    print(f"Dataset root: {root}\n")
    
    # Check data.yaml
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        print("✗ ERROR: data.yaml not found!")
        return False
    
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    num_classes = config.get('nc', 0)
    class_names = config.get('names', [])
    
    print(f"✓ data.yaml found")
    print(f"  Classes: {num_classes}")
    print(f"  Names defined: {len(class_names)}\n")
    
    if num_classes != len(class_names):
        print(f"⚠ WARNING: nc ({num_classes}) != len(names) ({len(class_names)})")
    
    # Validate each split
    results = {}
    for split in ['train', 'valid', 'test']:
        split_dir = root / split
        if not split_dir.exists():
            print(f"⚠ {split.upper()}: Not found (optional for test)")
            continue
        
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"✗ {split.upper()}: Missing images/ or labels/ folder")
            continue
        
        # Count files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        label_files = list(labels_dir.glob('*.txt'))
        
        # Match images to labels
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}
        
        matched = image_stems & label_stems
        images_only = image_stems - label_stems
        labels_only = label_stems - image_stems
        
        # Parse labels for class distribution
        class_counts = Counter()
        annotation_errors = []
        
        for label_file in label_files:
            if label_file.stem not in matched:
                continue
                
            try:
                with open(label_file) as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            annotation_errors.append(f"{label_file.name}:{line_num} - Invalid format")
                            continue
                        
                        class_id = int(parts[0])
                        if class_id >= num_classes:
                            annotation_errors.append(f"{label_file.name}:{line_num} - Class {class_id} out of range")
                            continue
                        
                        class_counts[class_id] += 1
            except Exception as e:
                annotation_errors.append(f"{label_file.name} - {str(e)}")
        
        # Check image integrity
        corrupted_images = []
        for img_file in list(image_files)[:10]:  # Sample first 10
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    corrupted_images.append(img_file.name)
            except:
                corrupted_images.append(img_file.name)
        
        # Report
        print(f"\n{split.upper()} SPLIT:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")
        print(f"  Matched pairs: {len(matched)}")
        
        if images_only:
            print(f"  ⚠ Images without labels: {len(images_only)}")
        if labels_only:
            print(f"  ⚠ Labels without images: {len(labels_only)}")
        
        if corrupted_images:
            print(f"  ⚠ Corrupted images (sample): {corrupted_images}")
        
        if annotation_errors:
            print(f"  ⚠ Annotation errors: {len(annotation_errors)} (showing first 5)")
            for err in annotation_errors[:5]:
                print(f"    - {err}")
        
        # Class distribution
        if class_counts:
            print(f"  Total annotations: {sum(class_counts.values())}")
            print(f"  Classes with data: {len(class_counts)}/{num_classes}")
            
            # Find empty classes
            empty_classes = set(range(num_classes)) - set(class_counts.keys())
            if empty_classes:
                print(f"  ⚠ Empty classes ({len(empty_classes)}): {sorted(empty_classes)[:10]}...")
        
        results[split] = {
            'images': len(image_files),
            'labels': len(label_files),
            'matched': len(matched),
            'class_counts': class_counts,
            'errors': len(annotation_errors) + len(corrupted_images)
        }
    
    # Summary
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION SUMMARY")
    print("="*60)
    
    if 'train' in results:
        train_classes = results['train']['class_counts']
        
        # Sort by frequency
        sorted_classes = sorted(train_classes.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 classes by frequency:")
        for class_id, count in sorted_classes[:10]:
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            print(f"  {class_name}: {count} annotations")
        
        print(f"\nBottom 10 classes by frequency:")
        for class_id, count in sorted_classes[-10:]:
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            print(f"  {class_name}: {count} annotations")
    
    # Indian-specific ingredient check
    print("\n" + "="*60)
    print("INDIAN INGREDIENT COVERAGE")
    print("="*60)
    
    indian_must_have = {
        'Paneer': False,
        'Palak': False,
        'Dal/Lentils': False,
        'Rice': False,
        'Ghee': False,
        'Turmeric': False,
        'Cumin': False,
        'Garam Masala': False,
        'Coriander': False,
        'Ginger': False,
        'Garlic': False
    }
    
    for name in class_names:
        name_lower = name.lower()
        if 'paneer' in name_lower:
            indian_must_have['Paneer'] = True
        if 'palak' in name_lower or 'spinach' in name_lower:
            indian_must_have['Palak'] = True
        if 'lentil' in name_lower or 'daal' in name_lower or 'dal' in name_lower:
            indian_must_have['Dal/Lentils'] = True
        if 'rice' in name_lower or 'chamal' in name_lower:
            indian_must_have['Rice'] = True
        if 'ghee' in name_lower:
            indian_must_have['Ghee'] = True
        if 'turmeric' in name_lower or 'haldi' in name_lower:
            indian_must_have['Turmeric'] = True
        if 'cumin' in name_lower or 'jeera' in name_lower:
            indian_must_have['Cumin'] = True
        if 'garam' in name_lower or 'masala' in name_lower:
            indian_must_have['Garam Masala'] = True
        if 'coriander' in name_lower or 'dhaniya' in name_lower:
            indian_must_have['Coriander'] = True
        if 'ginger' in name_lower or 'adrak' in name_lower:
            indian_must_have['Ginger'] = True
        if 'garlic' in name_lower or 'lehsun' in name_lower:
            indian_must_have['Garlic'] = True
    
    print("\nCritical Indian ingredients:")
    for ingredient, present in indian_must_have.items():
        status = "✓" if present else "✗ MISSING"
        print(f"  {status} {ingredient}")
    
    missing_count = sum(1 for v in indian_must_have.values() if not v)
    print(f"\nMissing critical ingredients: {missing_count}/11")
    
    if missing_count > 0:
        print("\n⚠ You need to capture images for missing ingredients!")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_roboflow.py <path_to_dataset>")
        print("\nExample:")
        print("  python validate_roboflow.py C:/Users/Tushar/Desktop/bawarchi/data/roboflow")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    validate_roboflow_dataset(dataset_path)