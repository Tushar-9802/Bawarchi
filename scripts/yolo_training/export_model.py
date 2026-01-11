#!/usr/bin/env python3
"""
BAWARCHI - Model Export Utility
Export trained YOLOv8 model to various formats for deployment
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def export_model(model_path, formats, output_dir, imgsz, half, simplify):
    """Export model to specified formats"""
    print("=" * 70)
    print("MODEL EXPORT UTILITY")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    print("Model loaded successfully")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_path}")
    print(f"Image size: {imgsz}")
    print(f"Half precision: {half}")
    print(f"Simplify (ONNX): {simplify}")
    
    # Export to each format
    print("\n" + "=" * 70)
    print("EXPORTING MODELS")
    print("=" * 70)
    
    export_results = {}
    
    for fmt in formats:
        print(f"\nExporting to {fmt.upper()}...")
        
        try:
            if fmt == 'torchscript':
                result = model.export(
                    format='torchscript',
                    imgsz=imgsz,
                    half=half
                )
            elif fmt == 'onnx':
                result = model.export(
                    format='onnx',
                    imgsz=imgsz,
                    half=half,
                    simplify=simplify,
                    dynamic=False
                )
            elif fmt == 'openvino':
                result = model.export(
                    format='openvino',
                    imgsz=imgsz,
                    half=half
                )
            elif fmt == 'coreml':
                result = model.export(
                    format='coreml',
                    imgsz=imgsz,
                    half=half
                )
            elif fmt == 'tflite':
                result = model.export(
                    format='tflite',
                    imgsz=imgsz,
                    half=half
                )
            else:
                print(f"Unsupported format: {fmt}")
                continue
            
            export_results[fmt] = result
            print(f"Success: {fmt.upper()} export complete")
            print(f"  Output: {result}")
            
        except Exception as e:
            print(f"Error exporting to {fmt.upper()}: {e}")
            export_results[fmt] = None
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    
    successful = [fmt for fmt, result in export_results.items() if result is not None]
    failed = [fmt for fmt, result in export_results.items() if result is None]
    
    if successful:
        print(f"\nSuccessful exports ({len(successful)}):")
        for fmt in successful:
            print(f"  - {fmt.upper()}")
    
    if failed:
        print(f"\nFailed exports ({len(failed)}):")
        for fmt in failed:
            print(f"  - {fmt.upper()}")
    
    print("\n" + "=" * 70)
    
    return export_results


def main():
    """Main export function"""
    parser = argparse.ArgumentParser(description='Export YOLOv8 Model')
    
    parser.add_argument('--model', type=str, default='runs/detect/bawarchi_baseline/weights/best.pt',
                       help='Path to trained model weights')
    parser.add_argument('--formats', type=str, nargs='+',
                       default=['torchscript', 'onnx'],
                       choices=['torchscript', 'onnx', 'openvino', 'coreml', 'tflite'],
                       help='Export formats')
    parser.add_argument('--output-dir', type=str, default='models/detection/weights',
                       help='Output directory for exported models')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for export')
    parser.add_argument('--half', action='store_true',
                       help='Use FP16 half precision')
    parser.add_argument('--simplify', action='store_true', default=True,
                       help='Simplify ONNX model')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Train model first: python scripts/yolo_training/train_baseline.py")
        return
    
    # Export model
    export_model(
        model_path=args.model,
        formats=args.formats,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        half=args.half,
        simplify=args.simplify
    )
    
    print("\nExport complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()