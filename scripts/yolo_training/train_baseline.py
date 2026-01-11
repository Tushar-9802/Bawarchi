#!/usr/bin/env python3
"""
BAWARCHI - Phase 2: YOLOv8 Baseline Training
Trains ingredient detection model on merged dataset (9,933 images, 124 classes)
"""

import argparse
from pathlib import Path
from datetime import datetime
import yaml
from ultralytics import YOLO


def verify_dataset(data_yaml_path):
    """Verify dataset exists and is properly configured"""
    print("=" * 70)
    print("DATASET VERIFICATION")
    print("=" * 70)
    
    if not Path(data_yaml_path).exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
    
    # Load and validate data.yaml
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nDataset configuration:")
    print(f"  Path: {data_yaml_path}")
    print(f"  Classes: {config.get('nc', 'NOT FOUND')}")
    print(f"  Train: {config.get('train', 'NOT FOUND')}")
    print(f"  Val: {config.get('val', 'NOT FOUND')}")
    print(f"  Test: {config.get('test', 'NOT FOUND')}")
    
    # Verify image directories exist
    base_path = Path(data_yaml_path).parent
    train_path = base_path / config.get('train', '')
    val_path = base_path / config.get('val', '')
    test_path = base_path / config.get('test', '')
    
    print(f"\nVerifying image directories...")
    for split, path in [('Train', train_path), ('Val', val_path), ('Test', test_path)]:
        if path.exists():
            img_count = len(list(path.glob('*.jpg'))) + len(list(path.glob('*.png')))
            print(f"  {split}: {img_count} images")
        else:
            print(f"  {split}: NOT FOUND")
    
    print("\n" + "=" * 70)
    return config


def setup_training_config(args):
    """Configure training parameters"""
    config = {
        'model': args.model,
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'patience': args.patience,
        'save_period': args.save_period,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'save': True,
        'save_json': False,
        'save_hybrid': False,
        'conf': None,
        'iou': 0.7,
        'max_det': 300,
        'half': False,
        'dnn': False,
        'workers': 8,
        'cache': False,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'crop_fraction': 1.0,
    }
    
    return config


def train_yolo(config):
    """Execute YOLOv8 training"""
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    
    print(f"\nModel: {config['model']}")
    print(f"Dataset: {config['data']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Image size: {config['imgsz']}")
    print(f"Batch size: {config['batch']}")
    print(f"Device: {config['device']}")
    print(f"Early stopping patience: {config['patience']}")
    print(f"Output: {config['project']}/{config['name']}")
    
    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)
    
    # Load pretrained YOLOv8 model
    model = YOLO(config['model'])
    
    print(f"\nLoaded: {config['model']}")
    print("Starting training...")
    
    # Start training
    print("\n" + "=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Monitor progress in terminal and TensorBoard")
    print(f"TensorBoard: tensorboard --logdir {config['project']}/{config['name']}")
    print("=" * 70 + "\n")
    
    results = model.train(**config)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results, model


def evaluate_model(model, config):
    """Run validation on test set"""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Validate on test set
    print("\nRunning validation on test set...")
    
    # Load data.yaml to get test path
    with open(config['data'], 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Run validation
    metrics = model.val(
        data=config['data'],
        split='test',
        batch=config['batch'],
        imgsz=config['imgsz'],
        device=config['device'],
        plots=True,
        save_json=True,
        conf=0.001,
        iou=0.6,
        max_det=300,
        half=False,
        dnn=False,
        workers=8,
        verbose=True
    )
    
    # Display key metrics
    print("\n" + "=" * 70)
    print("TEST SET METRICS")
    print("=" * 70)
    
    print(f"\nmAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    # Per-class metrics
    print("\n" + "=" * 70)
    print("PER-CLASS METRICS (Top 10 and Bottom 10)")
    print("=" * 70)
    
    # Get class names
    class_names = list(data_config.get('names', []))
    
    # Get per-class AP
    class_ap = metrics.box.ap50
    
    if len(class_ap) > 0 and len(class_names) > 0:
        # Sort classes by AP
        class_scores = [(name, ap) for name, ap in zip(class_names, class_ap)]
        class_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 classes:")
        for i, (name, ap) in enumerate(class_scores[:10], 1):
            print(f"  {i}. {name}: {ap:.3f}")
        
        print("\nBottom 10 classes:")
        for i, (name, ap) in enumerate(class_scores[-10:], 1):
            print(f"  {i}. {name}: {ap:.3f}")
    
    print("\n" + "=" * 70)
    
    return metrics


def save_training_summary(config, results, metrics, output_path):
    """Save training summary to text file"""
    summary_path = Path(output_path) / "training_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BAWARCHI - Phase 2 Training Summary\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Model: {config['model']}\n")
        f.write(f"Dataset: {config['data']}\n")
        f.write(f"Epochs: {config['epochs']}\n")
        f.write(f"Image size: {config['imgsz']}\n")
        f.write(f"Batch size: {config['batch']}\n")
        f.write(f"Device: {config['device']}\n\n")
        
        f.write("RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
        f.write(f"Precision: {metrics.box.mp:.4f}\n")
        f.write(f"Recall: {metrics.box.mr:.4f}\n\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"\nTraining summary saved: {summary_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='YOLOv8 Ingredient Detection Training')
    
    # Data parameters
    parser.add_argument('--data', type=str, default='data/merged/data.yaml',
                       help='Path to data.yaml')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLOv8 model size')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device (0, 1, 2, ...) or cpu')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=-1,
                       help='Save checkpoint every x epochs (-1 for no periodic saving)')
    
    # Output parameters
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='bawarchi_baseline',
                       help='Experiment name')
    parser.add_argument('--exist-ok', action='store_true',
                       help='Overwrite existing experiment')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BAWARCHI - PHASE 2: YOLOV8 BASELINE TRAINING")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Verify dataset
    data_config = verify_dataset(args.data)
    
    # Setup training configuration
    config = setup_training_config(args)
    
    # Train model
    results, model = train_yolo(config)
    
    # Evaluate on test set
    metrics = evaluate_model(model, config)
    
    # Save summary
    output_path = Path(config['project']) / config['name']
    save_training_summary(config, results, metrics, output_path)
    
    # Final output
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"\nModel weights saved:")
    print(f"  Best: {output_path}/weights/best.pt")
    print(f"  Last: {output_path}/weights/last.pt")
    print(f"\nTraining results:")
    print(f"  Location: {output_path}")
    print(f"  Summary: {output_path}/training_summary.txt")
    print(f"\nNext steps:")
    print(f"  1. Review training curves: {output_path}/results.png")
    print(f"  2. Check confusion matrix: {output_path}/confusion_matrix.png")
    print(f"  3. Test inference: python scripts/yolo_training/test_inference.py")
    print(f"  4. If mAP@0.5 < 0.60, proceed to Phase 3 optimization")
    print(f"  5. If mAP@0.5 >= 0.60, proceed to Phase 4 substitution learning")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()