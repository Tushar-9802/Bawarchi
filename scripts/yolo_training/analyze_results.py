#!/usr/bin/env python3
"""
BAWARCHI - Training Results Analysis
Analyze YOLOv8 training results and identify weak classes
"""

import argparse
from pathlib import Path
import pandas as pd
import yaml


def load_results(results_path):
    """Load training results CSV"""
    csv_path = Path(results_path) / "results.csv"
    
    if not csv_path.exists():
        print(f"Error: results.csv not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    return df


def analyze_training_curves(df):
    """Analyze training loss curves"""
    print("=" * 70)
    print("TRAINING CURVE ANALYSIS")
    print("=" * 70)
    
    # Get final epoch metrics
    final_metrics = df.iloc[-1]
    
    print(f"\nFinal epoch: {int(final_metrics['epoch'])}")
    print(f"\nLoss metrics:")
    print(f"  Box loss (train): {final_metrics.get('train/box_loss', 0):.4f}")
    print(f"  Class loss (train): {final_metrics.get('train/cls_loss', 0):.4f}")
    print(f"  DFL loss (train): {final_metrics.get('train/dfl_loss', 0):.4f}")
    
    print(f"\nValidation metrics:")
    print(f"  mAP@0.5: {final_metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP@0.5:0.95: {final_metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision: {final_metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall: {final_metrics.get('metrics/recall(B)', 0):.4f}")
    
    # Check for convergence
    print(f"\nConvergence analysis:")
    
    if len(df) >= 10:
        # Compare last 5 epochs to previous 5
        recent_map = df['metrics/mAP50(B)'].iloc[-5:].mean()
        previous_map = df['metrics/mAP50(B)'].iloc[-10:-5].mean()
        improvement = recent_map - previous_map
        
        print(f"  Recent mAP (last 5 epochs): {recent_map:.4f}")
        print(f"  Previous mAP (5 epochs before): {previous_map:.4f}")
        print(f"  Improvement: {improvement:.4f}")
        
        if improvement < 0.005:
            print("  Status: Converged (minimal improvement)")
        else:
            print("  Status: Still improving")
    
    print("\n" + "=" * 70)


def identify_weak_classes(results_path, data_yaml_path, threshold=0.4):
    """Identify classes with low precision"""
    print("\nWEAK CLASS IDENTIFICATION")
    print("=" * 70)
    
    # Load data.yaml for class names
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = config.get('names', [])
    
    # Look for per-class results
    # Note: Ultralytics saves per-class metrics in separate files during validation
    print(f"\nAnalyzing classes with precision < {threshold}")
    print("Note: Run validation with save_json=True to get detailed per-class metrics")
    
    # Check if confusion matrix exists
    confusion_matrix_path = Path(results_path) / "confusion_matrix.png"
    if confusion_matrix_path.exists():
        print(f"\nConfusion matrix available: {confusion_matrix_path}")
        print("Review visually to identify confused classes")
    
    # Check for results JSON
    results_json = Path(results_path) / "predictions.json"
    if results_json.exists():
        print(f"\nDetailed results available: {results_json}")
    else:
        print("\nRun validation to generate detailed metrics:")
        print("  python scripts/yolo_training/test_inference.py --source test")
    
    print("\n" + "=" * 70)


def check_data_distribution(data_yaml_path):
    """Check class distribution in dataset"""
    print("\nDATASET DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Load data config
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = Path(data_yaml_path).parent
    train_labels = base_path / 'train' / 'labels'
    
    if not train_labels.exists():
        # Try alternative structure
        train_labels = base_path / 'labels' / 'train'
    
    if not train_labels.exists():
        print("Warning: Label directory not found")
        return
    
    # Count class occurrences
    class_counts = {}
    label_files = list(train_labels.glob('*.txt'))
    
    print(f"\nAnalyzing {len(label_files)} label files...")
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls = int(parts[0])
                    class_counts[cls] = class_counts.get(cls, 0) + 1
    
    # Get class names
    class_names = config.get('names', [])
    
    # Sort by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTotal classes with annotations: {len(class_counts)}")
    print(f"Total annotations: {sum(class_counts.values())}")
    
    print("\nTop 10 most frequent classes:")
    for i, (cls_id, count) in enumerate(sorted_classes[:10], 1):
        name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
        print(f"  {i}. {name}: {count} annotations")
    
    print("\nBottom 10 least frequent classes:")
    for i, (cls_id, count) in enumerate(sorted_classes[-10:], 1):
        name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
        print(f"  {i}. {name}: {count} annotations")
    
    # Identify severely underrepresented classes
    severely_low = [(cls_id, count) for cls_id, count in class_counts.items() if count < 10]
    if severely_low:
        print(f"\nClasses with < 10 annotations ({len(severely_low)} classes):")
        for cls_id, count in sorted(severely_low, key=lambda x: x[1]):
            name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
            print(f"  - {name}: {count} annotations")
    
    print("\n" + "=" * 70)


def generate_recommendations(results_path, data_yaml_path):
    """Generate recommendations for Phase 3"""
    print("\nPHASE 3 RECOMMENDATIONS")
    print("=" * 70)
    
    # Load results
    df = load_results(results_path)
    
    if df is None:
        return
    
    final_map = df['metrics/mAP50(B)'].iloc[-1]
    
    print(f"\nCurrent mAP@0.5: {final_map:.4f}")
    
    if final_map >= 0.75:
        print("\nStatus: EXCELLENT - Exceeds target")
        print("Recommendations:")
        print("  1. Proceed directly to Phase 4 (Substitution Learning)")
        print("  2. Consider this model production-ready")
        print("  3. Test on real-world images to validate performance")
    
    elif final_map >= 0.60:
        print("\nStatus: GOOD - Meets baseline target")
        print("Recommendations:")
        print("  1. Proceed to Phase 4 (Substitution Learning)")
        print("  2. Optional: Minor optimization in Phase 3 for edge cases")
        print("  3. Monitor weak classes during integration testing")
    
    elif final_map >= 0.50:
        print("\nStatus: ACCEPTABLE - Needs optimization")
        print("Recommendations:")
        print("  1. Proceed to Phase 3 (Model Optimization)")
        print("  2. Focus on:")
        print("     - Hyperparameter tuning (learning rate, augmentation)")
        print("     - Weak class improvement (targeted data collection)")
        print("     - Consider YOLOv8s (small) instead of YOLOv8n (nano)")
        print("  3. Target: 65%+ mAP@0.5")
    
    else:
        print("\nStatus: NEEDS IMPROVEMENT - Major optimization required")
        print("Recommendations:")
        print("  1. Investigate data quality issues")
        print("  2. Check for:")
        print("     - Annotation errors in dataset")
        print("     - Class imbalance problems")
        print("     - Insufficient training data for some classes")
        print("  3. Consider:")
        print("     - Switch to YOLOv8s or YOLOv8m")
        print("     - Increase training epochs to 100")
        print("     - Adjust class weights")
        print("     - Collect more training data")
    
    print("\n" + "=" * 70)


def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Analyze YOLOv8 Training Results')
    
    parser.add_argument('--results', type=str, default='runs/detect/bawarchi_baseline',
                       help='Path to training results directory')
    parser.add_argument('--data', type=str, default='data/merged/data.yaml',
                       help='Path to data.yaml')
    parser.add_argument('--weak-threshold', type=float, default=0.4,
                       help='Threshold for identifying weak classes')
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_path}")
        print("Train model first: python scripts/yolo_training/train_baseline.py")
        return
    
    print("=" * 70)
    print("BAWARCHI - TRAINING RESULTS ANALYSIS")
    print("=" * 70)
    print(f"\nResults directory: {results_path}")
    
    # Load and analyze results
    df = load_results(results_path)
    
    if df is not None:
        # Analyze training curves
        analyze_training_curves(df)
        
        # Identify weak classes
        identify_weak_classes(results_path, args.data, args.weak_threshold)
        
        # Check data distribution
        check_data_distribution(args.data)
        
        # Generate recommendations
        generate_recommendations(results_path, args.data)
    
    print("\nAnalysis complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()