#!/usr/bin/env python3
"""
BAWARCHI - Dataset Merge Script
Combines Roboflow (120 classes) + Indian Spices (5 classes) into unified dataset
Handles class index remapping to avoid conflicts
"""

import shutil
from pathlib import Path
import yaml
from collections import defaultdict

# Paths
BASE_DIR = Path("C:/Users/Tushar/Desktop/bawarchi")
ROBOFLOW_DIR = BASE_DIR / "data" / "public" / "roboflow"
SPICES_DIR = BASE_DIR / "data" / "public" / "indian_spices_annotated"
MERGED_DIR = BASE_DIR / "data" / "merged"

print("=" * 70)
print("BAWARCHI - DATASET MERGE")
print("=" * 70)


def load_yaml(path):
    """Load YAML file"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    """Save YAML file"""
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)


def remap_labels(label_file, class_offset):
    """
    Remap class indices in label file by adding offset
    YOLO format: class_id x_center y_center width height
    """
    lines = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Remap class index
                old_class = int(parts[0])
                new_class = old_class + class_offset
                parts[0] = str(new_class)
                lines.append(' '.join(parts))
    return lines


def merge_datasets():
    """Main merge function"""
    
    # Load data.yaml files
    print("\n[1] LOADING DATASETS")
    print("-" * 70)
    
    roboflow_yaml = load_yaml(ROBOFLOW_DIR / "data.yaml")
    spices_yaml = load_yaml(SPICES_DIR / "data.yaml")
    
    roboflow_classes = roboflow_yaml['names']
    spices_classes = spices_yaml['names']
    
    print(f"Roboflow: {len(roboflow_classes)} classes")
    print(f"  Classes: {roboflow_classes[:10]}... (showing first 10)")
    print(f"\nIndian Spices: {len(spices_classes)} classes")
    print(f"  Classes: {spices_classes}")
    
    # Check for class name conflicts (case-insensitive)
    roboflow_lower = {c.lower(): i for i, c in enumerate(roboflow_classes)}
    conflicts = []
    spices_remap = {}  # Maps spices class index to merged class index
    
    for i, spice_class in enumerate(spices_classes):
        spice_lower = spice_class.lower()
        if spice_lower in roboflow_lower:
            # Conflict: spice class already exists in roboflow
            merged_idx = roboflow_lower[spice_lower]
            spices_remap[i] = merged_idx
            conflicts.append((spice_class, roboflow_classes[merged_idx]))
        else:
            # No conflict: append to end
            merged_idx = len(roboflow_classes) + len(spices_remap) - len(conflicts)
            spices_remap[i] = merged_idx
    
    # Build merged class list
    merged_classes = roboflow_classes.copy()
    for i, spice_class in enumerate(spices_classes):
        if spice_class.lower() not in roboflow_lower:
            merged_classes.append(spice_class)
    
    print(f"\n[2] CLASS MAPPING")
    print("-" * 70)
    if conflicts:
        print(f"⚠ Found {len(conflicts)} overlapping classes:")
        for spice, robo in conflicts:
            print(f"  '{spice}' → merged with '{robo}'")
    else:
        print("✓ No class conflicts")
    
    print(f"\nMerged dataset: {len(merged_classes)} total classes")
    print(f"  {len(roboflow_classes)} from Roboflow")
    print(f"  {len(merged_classes) - len(roboflow_classes)} new from Indian Spices")
    
    # Create merged directory structure
    print(f"\n[3] CREATING MERGED DATASET")
    print("-" * 70)
    
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'valid', 'test']
    for split in splits:
        (MERGED_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy Roboflow data (no remapping needed, class indices stay same)
    print("\n[3.1] Copying Roboflow dataset...")
    robo_stats = defaultdict(int)
    
    for split in splits:
        src_img_dir = ROBOFLOW_DIR / split / 'images'
        src_lbl_dir = ROBOFLOW_DIR / split / 'labels'
        dst_img_dir = MERGED_DIR / split / 'images'
        dst_lbl_dir = MERGED_DIR / split / 'labels'
        
        if src_img_dir.exists():
            images = list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png"))
            
            for img in images:
                # Copy image
                shutil.copy2(img, dst_img_dir / img.name)
                
                # Copy label
                label_file = src_lbl_dir / f"{img.stem}.txt"
                if label_file.exists():
                    shutil.copy2(label_file, dst_lbl_dir / label_file.name)
            
            robo_stats[split] = len(images)
            print(f"  {split}: {len(images)} images")
    
    # Copy and remap Indian Spices data
    print("\n[3.2] Copying and remapping Indian Spices...")
    spice_stats = defaultdict(int)
    
    for split in splits:
        src_img_dir = SPICES_DIR / split / 'images'
        src_lbl_dir = SPICES_DIR / split / 'labels'
        dst_img_dir = MERGED_DIR / split / 'images'
        dst_lbl_dir = MERGED_DIR / split / 'labels'
        
        if src_img_dir.exists():
            images = list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png"))
            
            for img in images:
                # Generate unique name to avoid conflicts
                new_name = f"spice_{img.name}"
                
                # Copy image
                shutil.copy2(img, dst_img_dir / new_name)
                
                # Remap and copy label
                label_file = src_lbl_dir / f"{img.stem}.txt"
                if label_file.exists():
                    # Read and remap labels
                    remapped_lines = []
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                old_class = int(parts[0])
                                new_class = spices_remap[old_class]
                                parts[0] = str(new_class)
                                remapped_lines.append(' '.join(parts))
                    
                    # Write remapped labels
                    new_label = dst_lbl_dir / f"{Path(new_name).stem}.txt"
                    with open(new_label, 'w') as f:
                        f.write('\n'.join(remapped_lines))
            
            spice_stats[split] = len(images)
            print(f"  {split}: {len(images)} images")
    
    # Create merged data.yaml
    print("\n[4] CREATING data.yaml")
    print("-" * 70)
    
    merged_yaml = {
        'path': str(MERGED_DIR.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(merged_classes),
        'names': merged_classes
    }
    
    save_yaml(merged_yaml, MERGED_DIR / "data.yaml")
    
    print(f"✓ Created: {MERGED_DIR / 'data.yaml'}")
    print(f"  Classes: {len(merged_classes)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    
    total_train = robo_stats['train'] + spice_stats['train']
    total_valid = robo_stats['valid'] + spice_stats['valid']
    total_test = robo_stats['test'] + spice_stats['test']
    total_images = total_train + total_valid + total_test
    
    print(f"\n[FINAL DATASET STATS]")
    print(f"Location: {MERGED_DIR}")
    print(f"Total classes: {len(merged_classes)}")
    print(f"Total images: {total_images:,}")
    print(f"  Train: {total_train:,} ({robo_stats['train']:,} roboflow + {spice_stats['train']:,} spices)")
    print(f"  Valid: {total_valid:,} ({robo_stats['valid']:,} roboflow + {spice_stats['valid']:,} spices)")
    print(f"  Test: {total_test:,} ({robo_stats['test']:,} roboflow + {spice_stats['test']:,} spices)")
    
    print(f"\n[CLASS LIST]")
    print(f"First 10 classes: {merged_classes[:10]}")
    print(f"Last 10 classes: {merged_classes[-10:]}")
    
    print(f"\n[NEXT STEPS]")
    print(f"1. Verify merged dataset: python scripts/validate_merged.py")
    print(f"2. Custom capture: Ghee, Turmeric, Garam Masala (45 images)")
    print(f"3. Start training: Week 2 - YOLOv8 baseline")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    merge_datasets()