#!/usr/bin/env python3
"""
Create YOLO labels from Open Images annotations
"""

import csv
from pathlib import Path

metadata_dir = Path("data/public/open_images/metadata")
annotations_file = metadata_dir / "train-annotations-bbox.csv"
images_dir = Path("data/public/open_images/images")
labels_dir = Path("data/public/open_images/labels")

labels_dir.mkdir(parents=True, exist_ok=True)

# Class mapping
classes = ["apple", "banana", "carrot", "tomato", "orange", "broccoli", "chicken"]

# Load class IDs from metadata
class_id_to_idx = {}
classes_file = metadata_dir / "class-descriptions.csv"

with open(classes_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) != 2:
            continue
        class_id, class_name = row
        class_name_lower = class_name.lower()
        
        if class_name_lower in classes:
            class_id_to_idx[class_id] = classes.index(class_name_lower)

print(f"Creating YOLO labels for {len(class_id_to_idx)} classes...")

# Get list of downloaded images
downloaded_images = {img.stem for img in images_dir.glob("*.jpg")}
print(f"Found {len(downloaded_images)} downloaded images")

# Process annotations
annotations = {}  # image_id -> list of boxes

print("Reading annotations...")
with open(annotations_file, 'r') as f:
    reader = csv.DictReader(f)
    
    for i, row in enumerate(reader):
        if i % 100000 == 0 and i > 0:
            print(f"  Processed {i:,} annotations...")
        
        image_id = row.get('ImageID')
        
        # Only process if we downloaded this image
        if image_id not in downloaded_images:
            continue
        
        label_name = row.get('LabelName')
        
        # Only process if this is a class we want
        if label_name not in class_id_to_idx:
            continue
        
        class_idx = class_id_to_idx[label_name]
        
        # Get bounding box (in normalized coordinates)
        xmin = float(row.get('XMin'))
        ymin = float(row.get('YMin'))
        xmax = float(row.get('XMax'))
        ymax = float(row.get('YMax'))
        
        # Convert to YOLO format (center_x, center_y, width, height)
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        
        if image_id not in annotations:
            annotations[image_id] = []
        
        annotations[image_id].append([class_idx, x_center, y_center, width, height])

print(f"\n✓ Processed {len(annotations)} images with annotations")

# Write YOLO label files
print("Writing YOLO label files...")
for image_id, boxes in annotations.items():
    label_file = labels_dir / f"{image_id}.txt"
    
    with open(label_file, 'w') as f:
        for box in boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

print(f"✓ Created {len(annotations)} label files")

# Create data.yaml
yaml_content = f"""path: {Path('data/public/open_images').absolute()}
train: images
val: images

nc: {len(classes)}
names: {classes}
"""

yaml_file = Path("data/public/open_images/data.yaml")
with open(yaml_file, 'w') as f:
    f.write(yaml_content)

print(f"✓ Created {yaml_file}")

print("\n" + "=" * 60)
print("✓ Open Images dataset ready!")
print("=" * 60)
print(f"Images: {len(downloaded_images)}")
print(f"Labels: {len(annotations)}")
print(f"Classes: {len(classes)}")