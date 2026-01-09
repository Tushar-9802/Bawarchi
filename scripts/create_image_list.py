#!/usr/bin/env python3
"""
Create image list for Open Images downloader
Filters for specific food classes
"""

import csv
from pathlib import Path
from urllib.request import urlretrieve

# Download metadata files first
metadata_dir = Path("data/public/open_images/metadata")
metadata_dir.mkdir(parents=True, exist_ok=True)

print("Downloading metadata files...")

# Class descriptions
classes_url = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
classes_file = metadata_dir / "class-descriptions.csv"

if not classes_file.exists():
    print("  Downloading class descriptions...")
    urlretrieve(classes_url, classes_file)
    print("  ✓ Downloaded")

# Train annotations
annotations_url = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
annotations_file = metadata_dir / "train-annotations-bbox.csv"

if not annotations_file.exists():
    print("  Downloading train annotations (this is large, ~1.5 GB)...")
    urlretrieve(annotations_url, annotations_file)
    print("  ✓ Downloaded")

print("\nMapping class names to IDs...")

# Target classes
target_classes = {
    "apple": None,
    "banana": None,
    "carrot": None,
    "tomato": None,
    "orange": None,
    "broccoli": None,
    "chicken": None,
    "corn": None,
    "potato": None,

}

# Map class names to IDs
with open(classes_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) != 2:
            continue
        class_id, class_name = row
        class_name_lower = class_name.lower()
        
        if class_name_lower in target_classes:
            target_classes[class_name_lower] = class_id
            print(f"  {class_name}: {class_id}")

# Filter found classes
found_classes = {name: id for name, id in target_classes.items() if id is not None}
print(f"\n✓ Found {len(found_classes)}/{len(target_classes)} classes")

if len(found_classes) < 5:
    print("⚠️  Warning: Many classes not found")
    exit(1)

print("\nFiltering annotations...")
print("This may take 5-10 minutes...")

# Collect image IDs
image_ids = set()
max_per_class = 1200  # Target per class

class_counts = {name: 0 for name in found_classes.keys()}

with open(annotations_file, 'r') as f:
    reader = csv.DictReader(f)
    
    for i, row in enumerate(reader):
        if i % 100000 == 0 and i > 0:
            print(f"  Processed {i:,} annotations...")
        
        label_name = row.get('LabelName')
        image_id = row.get('ImageID')
        
        # Check if this label is one we want
        for class_name, class_id in found_classes.items():
            if label_name == class_id and class_counts[class_name] < max_per_class:
                image_ids.add(f"train/{image_id}")
                class_counts[class_name] += 1
        
        # Stop if we have enough
        if all(count >= max_per_class for count in class_counts.values()):
            break

print(f"\n✓ Found {len(image_ids):,} unique images")
print("\nImages per class:")
for class_name, count in class_counts.items():
    print(f"  {class_name}: {count}")

# Save image list
output_file = Path("scripts/image_list.txt")
with open(output_file, 'w') as f:
    for img_id in sorted(image_ids):
        f.write(f"{img_id}\n")

print(f"\n✓ Saved image list: {output_file}")
print(f"\nNext step:")
print(f"  python scripts/downloader.py scripts/image_list.txt --download_folder=data/public/open_images/images --num_processes=5")