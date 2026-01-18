# Bawarchi

**AI-powered ingredient detection and fusion recipe generation**

8-phase ML system combining computer vision (YOLOv8) and natural language processing (Llama 3.2 3B) to detect ingredients from images and generate Indian-Mexican fusion recipes with automated substitution learning.

## Architecture

```
Image Input -> YOLOv8m Detection -> Ingredient List -> 
Substitution Network -> Llama 3.2 3B Generation -> Fusion Recipe
```

**Components:**

1. **YOLOv8m** - Ingredient detection (124 classes, 66.51% mAP@0.5)
2. **Substitution Network** - Automated ingredient substitution learning (embedding-based)
3. **Llama 3.2 3B + LoRA** - Recipe generation with fusion logic (Windows compatible)
4. **Cultural Context** - Indian/Mexican cuisine classification

---

## Current Status

**Completed:**

- [X] Phase 1: Dataset acquisition and validation
- [X] Phase 2: YOLOv8m detection model training

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
- Generation: Llama 3.2 3B + LoRA (3B params, bfloat16 precision)

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
- Training: LoRA (r=64, alpha=16, bfloat16 precision, Windows compatible)
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
│   ├── merged/                    # 7,229 detection images (YOLO)
│   ├── training/                  # 380K recipes (JSONL, excluded)
│   ├── substitution/              # PMI matrices, embeddings
│   └── recipes/food_com/          # 522K recipe corpus (parquet)
├── models/
│   ├── detection/
│   │   └── production/            # YOLOv8m weights (66.51% mAP)
│   ├── bawarchi-adapter/
│   │   ├── final/                 # Best LoRA adapter (Phase 3)
│   │   ├── phase_1/               # Checkpoints (simple recipes)
│   │   ├── phase_2/               # Checkpoints (complex recipes)
│   │   └── phase_3/               # Checkpoints (fusion recipes)
├── results/
│   ├── detection/                 # YOLOv8 training curves
│   └── recipe_generation/         # Llama training logs
├── scripts/
│   ├── detection/                 # Phase 2 scripts
│   ├── substitution_learning/     # Phase 3 scripts (5 modules)
│   └── recipe_generation/         # Phase 4 scripts (6 modules)
└── requirements.txt
```

---

## Known Limitations

**Detection:**

- Weak classes (<10 annotations): Orange, Yellow Lentils, Long Beans, etc.
- Missing ingredients: Ghee, Turmeric powder, Garam Masala
- Static images only (no video inference)

**Recipe Generation:**

- Partially trained (11h vs. optimal 24h)
- Quality: 74% vs. potential 90%+ with full training
- Limited to 2048 token context

**Dataset:**

- Limited to 124 ingredient classes
- Bias toward well-represented ingredients
- General cuisine over-represented (89.7% of recipes)

---

## Future Enhancements

- [ ] Complete recipe generation training (Phase 1-3 extended)
- [ ] Collect weak class data (+5-8% mAP improvement)
- [ ] Expand cuisine coverage (Thai, Mediterranean, Chinese)
- [ ] Mobile deployment (ONNX export)
- [ ] Real-time video inference
- [ ] Nutrition estimation
- [ ] User feedback loop

---

## Development Notes

**Phase 2 (Detection) Insights:**

- Larger models worth compute cost (+11% over baseline)
- Batch size critical for GPU utilization (32-64 optimal)
- Early stopping effective (converged at epoch 97/150)

**Phase 3 (Substitution) Insights:**

- PMI + embeddings complementary (RRF best)
- Category filtering essential (85% → 91% precision)
- Recipe corpus size matters (522K → good coverage)

**Phase 4 (Generation) Insights:**

- Curriculum learning crucial (simple → complex)
- Response template masking critical (+10-15% quality)
- Data loading bottleneck (workers=4 → 8x speedup)
- Evaluation expensive (reduced to 1000 steps)
- Quality filters important (43K/80K synthetic accepted)

---

## License

MIT License

---

## Author

**Tushar Jaju**
GitHub: [Tushar-9802/bawarchi](https://github.com/Tushar-9802/bawarchi)
