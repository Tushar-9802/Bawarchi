
# Bawarchi

**AI-powered ingredient detection and fusion recipe generation**

8-phase ML system combining computer vision (YOLOv8) and natural language processing (Llama 3.2 3B) to detect ingredients from images and generate Indian-Mexican fusion recipes with automated substitution learning.

## Architecture

```
Image Input → YOLOv8m Detection → Ingredient List → 
Substitution Network → Llama 3.2 3B Generation → Fusion Recipe
```

**Components:**

1. **YOLOv8m** - Ingredient detection (124 classes, 66.51% mAP@0.5)
2. **Substitution Network** - Automated ingredient substitution learning (embedding-based)
3. **Llama 3.2 3B + QLoRA** - Recipe generation with fusion logic
4. **Cultural Context** - Indian/Mexican cuisine classification

---

## Current Status

**Completed:**

- ✓ Phase 1: Dataset acquisition and validation
- ✓ Phase 2: YOLOv8m detection model training

**In Progress:**

- Phase 4: Substitution learning system

**Remaining:**

- Phase 5: Recipe generation (Llama 3.2 3B)
- Phase 6: End-to-end integration
- Phase 7: Missing ingredient suggestions
- Phase 8: Testing and deployment

---

## Datasets

| Dataset                           | Images          | Classes       | Purpose                            |
| --------------------------------- | --------------- | ------------- | ---------------------------------- |
| Roboflow Food-Ingredients         | 9,731           | 120           | Base ingredient detection          |
| Indian Spices (manual annotation) | 202             | 5             | Indian-specific ingredients        |
| **Merged Dataset**          | **9,933** | **124** | **Production training data** |
| Food.com Recipes                  | 54,857          | -             | Recipe generation corpus           |

**Recipe Distribution:**

- Indian: 22,026 recipes
- Mexican: 34,475 recipes
- Fusion overlap: 1,644 recipes

---

## Model Performance

**Detection Model (YOLOv8m):**

- mAP@0.5: 66.51% (6.51% above target)
- mAP@0.5:0.95: 41.10%
- Precision: 68.91%
- Recall: 59.14%
- Parameters: 25.9M
- Inference speed: ~30-40 FPS (RTX 5070 Ti)

**Training progression:**

- YOLOv8n baseline: 55.55% mAP@0.5
- YOLOv8s optimized: 62.75% mAP@0.5
- YOLOv8m production: 66.51% mAP@0.5

---

## Quick Start

**Prerequisites:**

- Python 3.11+
- CUDA-compatible GPU (16GB+ VRAM recommended)
- Windows/Linux

**Setup:**

```bash
# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python -c "import torch; print(torch.cuda.is_available())"
```

**Inference:**

```bash
# Run detection on image
python scripts/yolo_training/test_inference.py --source image.jpg --model models/detection/production/yolov8m_best.pt
```

---

## Technical Specifications

**Hardware:**

- GPU: NVIDIA RTX 5070 Ti (Blackwell architecture)
- VRAM: 16GB
- CUDA: 12.8
- PyTorch: 2.11.0.dev

**Models:**

- Detection: YOLOv8m (25.9M params)
- Generation: Llama 3.2 3B + QLoRA (3B params, 4-bit quantized)

**Training:**

- Detection: 100 epochs, batch 32-64, ~4 hours
- Generation: 3 epochs, batch 4, ~6-8 hours (planned)

---

## System Design

**Phase 4: Substitution Learning**

- Ingredient embeddings (visual + semantic, 960-dim)
- Co-occurrence graph from recipe corpus
- Contrastive learning for similarity
- Cooking method compatibility filtering
- Cultural context weighting (Indian/Mexican)

**Phase 5: Recipe Generation**

- Model: Llama 3.2 3B (Meta)
- Training: QLoRA (r=16, alpha=32, 4-bit quantization)
- Input: Detected ingredients + optional cuisine preference
- Output: Fusion recipe with instructions
- Success threshold: 4.0/5.0 human evaluation

**Substitution Approach:**

- Automated learning from recipe variations
- No manual taxonomy encoding
- Adaptive from user feedback
- Cooking method aware (fry/boil/roast compatibility)

---

## Project Structure

```
bawarchi/
├── data/
│   ├── merged/                    # 9,933 training images (YOLO format)
│   ├── recipes/                   # 54,857 recipe corpus (parquet)
│   └── substitution/              # Substitution learning data
├── models/
│   ├── detection/
│   │   └── production/            # YOLOv8m weights (66.51% mAP)
│   ├── substitution/              # Similarity networks
│   └── generation/                # Llama 3.2 3B checkpoints
├── results/
│   └── detection/
│       ├── production/            # Training curves, metrics
│       └── comparison/            # Model comparison data
├── scripts/
│   ├── dataset_preparation/       # Phase 1 scripts
│   ├── yolo_training/             # Phase 2 scripts
│   └── substitution_learning/     # Phase 4 scripts (in progress)
└── logs/
    └── detection/                 # Training logs
```

---

## Known Limitations

**Detection:**

- Weak classes (<10 annotations): Orange, Yellow Lentils, Long Beans, Water Melon, Ice, Strawberry, Wallnut, Green Peas, Crab Meat, Minced Meat
- Missing ingredients: Ghee, Turmeric powder, Garam Masala (require custom collection)

**Dataset:**

- Static images only (no video inference)
- Limited to 124 ingredient classes
- Bias toward well-represented ingredients

---

## Future Enhancements

- Collect weak class data (50-100 images per class) for +5-8% mAP improvement
- Expand cuisine coverage (Thai, Mediterranean, Chinese)
- Mobile deployment (ONNX export for edge devices)
- Real-time video inference
- Nutrition estimation
- User feedback loop for substitution refinement

---

## Development Notes

**Phase 2 Insights:**

- Larger models (YOLOv8m) worth the compute cost (+11% over baseline)
- Batch size optimization critical for GPU utilization (32-64 optimal)
- Early stopping effective (converged at epoch 97/150)
- Class imbalance major limiting factor (need targeted data collection)

**Phase 4 Challenges:**

- Ingredient name matching between detection classes and recipe text
- Recipe pair scarcity (need validation)
- Substitution quality validation (no ground truth)

---

## License

MIT License

---

## Author

**Tushar Jaju**

GitHub: [Tushar-9802/bawarchi](https://github.com/Tushar-9802/bawarchi)
