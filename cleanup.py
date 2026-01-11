"""
BAWARCHI - Detection Model Cleanup
Reorganizes YOLOv8 training artifacts and deletes redundant runs.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def copy_file(src, dst):
    """Copy file silently"""
    try:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            return True
    except:
        pass
    return False

def find_best_run(runs):
    """Select best run - prefer _v2 over _v22, must have weights"""
    if not runs:
        return None
    
    # Filter runs with actual model files
    valid = [r for r in runs if (r / "weights" / "best.pt").exists()]
    if not valid:
        return None
    
    # Prefer runs ending with _v2 (not _v22)
    v2_exact = [r for r in valid if r.name.endswith('_v2')]
    if v2_exact:
        return v2_exact[0]
    
    # Otherwise take most recent
    return sorted(valid, key=lambda x: x.stat().st_mtime)[-1]

def main():
    print("\nBAWARCHI - Cleanup Detection Training Artifacts\n")
    
    # Check directory
    if not os.path.exists("data") or not os.path.exists("models"):
        print("Error: Run from bawarchi/ root directory")
        return
    
    # Create structure
    print("Creating directories...")
    for d in ["models/detection/production", "models/detection/archived", 
              "logs/detection", "results/detection/production", 
              "results/detection/comparison"]:
        os.makedirs(d, exist_ok=True)
    
    # Find runs
    print("Identifying runs...")
    runs_dir = Path("runs/detect")
    if not runs_dir.exists():
        print("Error: runs/detect not found")
        return
    
    n_runs = [r for r in runs_dir.iterdir() if r.is_dir() and 'baseline' in r.name.lower()]
    s_runs = [r for r in runs_dir.iterdir() if r.is_dir() and 'optimized' in r.name.lower()]
    m_runs = [r for r in runs_dir.iterdir() if r.is_dir() and ('medium' in r.name.lower() or 'aggressive' in r.name.lower())]
    
    n_best = find_best_run(n_runs)
    s_best = find_best_run(s_runs)
    m_best = find_best_run(m_runs)
    
    print(f"  YOLOv8n: {n_best.name if n_best else 'N/A'}")
    print(f"  YOLOv8s: {s_best.name if s_best else 'N/A'}")
    print(f"  YOLOv8m: {m_best.name if m_best else 'N/A'}")
    
    if not m_best:
        print("\nError: YOLOv8m production run not found")
        return
    
    # Copy production model
    print("\nCopying production model...")
    m = m_best
    copy_file(m / "weights/best.pt", "models/detection/production/yolov8m_best.pt")
    copy_file(m / "weights/best.torchscript", "models/detection/production/yolov8m_best.torchscript")
    copy_file(m / "results.png", "results/detection/production/training_curves.png")
    copy_file(m / "confusion_matrix.png", "results/detection/production/confusion_matrix.png")
    copy_file(m / "results.csv", "results/detection/production/results.csv")
    copy_file(m / "training_summary.txt", "results/detection/production/training_summary.txt")
    copy_file(m / "labels.jpg", "results/detection/production/labels.jpg")
    
    # Model info
    info = f"""Production Detection Model
=========================
Model: YOLOv8m
Training run: {m.name}
Date: {datetime.now().strftime('%Y-%m-%d')}

Specifications:
  Architecture: YOLOv8m
  Parameters: 25.9M
  Performance: 66.51% mAP@0.5
  Dataset: 9,933 images, 124 classes

Files:
  PyTorch: models/detection/production/yolov8m_best.pt
  TorchScript: models/detection/production/yolov8m_best.torchscript
  Results: results/detection/production/
"""
    with open("models/detection/production/MODEL_INFO.txt", 'w', encoding='utf-8') as f:
        f.write(info)
    
    # Copy comparison data
    print("Copying comparison data...")
    if n_best:
        copy_file(n_best / "training_summary.txt", "results/detection/comparison/yolov8n_summary.txt")
        copy_file(n_best / "results.png", "results/detection/comparison/yolov8n_curves.png")
    if s_best:
        copy_file(s_best / "training_summary.txt", "results/detection/comparison/yolov8s_summary.txt")
        copy_file(s_best / "results.png", "results/detection/comparison/yolov8s_curves.png")
    
    # Extract mAP values
    def get_map(run):
        if not run:
            return "N/A"
        try:
            with open(run / "training_summary.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    if 'mAP@0.5:' in line:
                        return line.split('mAP@0.5:')[1].strip().split()[0]
        except:
            pass
        return "N/A"
    
    n_map = get_map(n_best)
    s_map = get_map(s_best)
    m_map = get_map(m_best)
    
    # Comparison file
    comp = f"""Model Comparison
================
Training date: {datetime.now().strftime('%Y-%m-%d')}

YOLOv8n (Baseline)
  Run: {n_best.name if n_best else 'N/A'}
  Params: 3.3M | mAP@0.5: {n_map}

YOLOv8s (Optimized)
  Run: {s_best.name if s_best else 'N/A'}
  Params: 11M | mAP@0.5: {s_map}

YOLOv8m (Production)
  Run: {m_best.name if m_best else 'N/A'}
  Params: 25.9M | mAP@0.5: {m_map}

Selection rationale: YOLOv8m selected for production
"""
    with open("results/detection/comparison/COMPARISON.txt", 'w', encoding='utf-8') as f:
        f.write(comp)
    
    # Delete runs
    print("\nDeleting runs directory...")
    total_size = 0
    try:
        for root, dirs, files in os.walk(runs_dir):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        
        shutil.rmtree("runs/detect")
        if os.path.exists("runs") and not os.listdir("runs"):
            os.rmdir("runs")
        
        print(f"  Freed: {total_size / (1024*1024):.1f} MB")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Update gitignore
    print("Updating .gitignore...")
    rules = """
# Detection models (large files)
models/detection/production/*.pt
models/detection/production/*.torchscript
models/detection/archived/*.pt
*.cache
logs/*.log
"""
    try:
        existing = ""
        if os.path.exists(".gitignore"):
            with open(".gitignore", 'r', encoding='utf-8') as f:
                existing = f.read()
        
        if "# Detection models" not in existing:
            with open(".gitignore", 'a', encoding='utf-8') as f:
                f.write(rules)
    except:
        pass
    
    # Report
    print("\nCleanup complete")
    print("\nVerify:")
    print("  ls models/detection/production/")
    print("  cat results/detection/comparison/COMPARISON.txt")
    print("\nDelete this script:")
    print("  rm cleanup_detection.py")

if __name__ == "__main__":
    main()