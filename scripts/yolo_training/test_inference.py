#!/usr/bin/env python3
"""
BAWARCHI - YOLOv8 Inference Testing
Test trained model on images and generate predictions
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2


def test_on_images(model_path, source, conf_threshold, save_dir, show_labels):
    """Run inference on images and save results"""
    print("=" * 70)
    print("YOLOV8 INFERENCE TESTING")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Model loaded successfully")
    print(f"Classes: {len(model.names)}")
    print(f"Source: {source}")
    print(f"Confidence threshold: {conf_threshold}")
    
    # Run inference
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70 + "\n")
    
    results = model.predict(
        source=source,
        conf=conf_threshold,
        save=True,
        save_txt=False,
        save_conf=True,
        project=save_dir,
        name='predictions',
        exist_ok=True,
        line_width=2,
        show_labels=show_labels,
        show_conf=True,
        verbose=True
    )
    
    # Analyze results
    print("\n" + "=" * 70)
    print("INFERENCE RESULTS")
    print("=" * 70)
    
    total_detections = 0
    detection_counts = {}
    
    for r in results:
        boxes = r.boxes
        img_name = Path(r.path).name
        
        if len(boxes) > 0:
            print(f"\n{img_name}: {len(boxes)} detections")
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                print(f"  - {class_name}: {conf:.3f}")
                
                total_detections += 1
                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
        else:
            print(f"\n{img_name}: No detections")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal images processed: {len(results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/len(results):.2f}")
    
    if detection_counts:
        print(f"\nDetected ingredient classes: {len(detection_counts)}")
        print("\nTop 10 detected ingredients:")
        sorted_detections = sorted(detection_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (ingredient, count) in enumerate(sorted_detections[:10], 1):
            print(f"  {i}. {ingredient}: {count} detections")
    
    output_path = Path(save_dir) / 'predictions'
    print(f"\nPredictions saved to: {output_path}")
    
    print("\n" + "=" * 70 + "\n")


def test_on_test_split(model_path, data_yaml, conf_threshold, save_dir):
    """Run inference on entire test split"""
    print("=" * 70)
    print("TEST SPLIT INFERENCE")
    print("=" * 70)
    
    import yaml
    
    # Load data config
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = Path(data_yaml).parent
    test_path = base_path / config.get('test', '')
    
    if not test_path.exists():
        print(f"\nError: Test split not found at {test_path}")
        return
    
    # Get test images
    test_images = list(test_path.glob('*.jpg')) + list(test_path.glob('*.png'))
    
    print(f"\nTest split: {len(test_images)} images")
    print(f"Location: {test_path}")
    
    # Run inference
    test_on_images(model_path, str(test_path), conf_threshold, save_dir, show_labels=True)


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='YOLOv8 Inference Testing')
    
    parser.add_argument('--model', type=str, default='runs/detect/bawarchi_baseline/weights/best.pt',
                       help='Path to trained model weights')
    parser.add_argument('--source', type=str, default='test',
                       help='Image source (file, folder, or "test" for test split)')
    parser.add_argument('--data', type=str, default='data/merged/data.yaml',
                       help='Path to data.yaml (required if source="test")')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--save-dir', type=str, default='runs/inference',
                       help='Directory to save results')
    parser.add_argument('--save-labels', action='store_true',
                       help='Save prediction labels to txt files')
    parser.add_argument('--show-labels', action='store_true', default=True,
                       help='Show labels on images')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Train model first: python scripts/yolo_training/train_baseline.py")
        return
    
    # Run inference
    if args.source == 'test':
        test_on_test_split(args.model, args.data, args.conf, args.save_dir)
    else:
        test_on_images(args.model, args.source, args.conf, args.save_dir, args.show_labels)


if __name__ == "__main__":
    main()