# Bawarchi

**AI-powered ingredient detection and fusion recipe generation**

Hybrid ML system combining computer vision (YOLOv8) and natural language processing (T5-LoRA) to detect ingredients from images and generate Indian-Mexican fusion recipes.

Architecture

```
Image Input → YOLOv8 Detection → Ingredient List → T5-LoRA Generation → Fusion Recipe
```

**Components:**

1. **YOLOv8** - Ingredient detection (124 classes)
2. **T5-base + LoRA** - Recipe generation with cultural fusion logic
3. **Substitution Engine** - Ingredient interchange rules (roti ↔ tortilla, paneer ↔ tofu)

---

## Datasets

| Dataset                   | Images          | Classes       | Purpose                     |
| ------------------------- | --------------- | ------------- | --------------------------- |
| Roboflow Food-Ingredients | 9,731           | 120           | Base ingredient detection   |
| Indian Spices (annotated) | 202             | 5             | Indian-specific ingredients |
| **Merged Dataset**  | **9,933** | **124** | **Training dataset**  |
| Food.com Recipes          | 54,857          | -             | Recipe generation corpus    |

**Recipe Breakdown:**

* Indian: 22,026 recipes
* Mexican: 34,475 recipes
* Fusion overlap: 1,644 recipes

---

## Quick Start

**Prerequisites:**

* Python 3.11.x
* CUDA-compatible GPU (tested on RTX 5070 Ti)
* 16GB+ VRAM recommended

**Setup:**

```bash
# Install dependencies
pip install -r requirements.txt

# Verify GPU
python scripts/verify_gpu.py

# Train YOLOv8 (Week 2)
python scripts/train_yolo.py --data data/merged/data.yaml --epochs 50

# Test model
python scripts/test_model.py --source test_images/
```

---

## Technical Specs

**Training Hardware:**

* GPU: NVIDIA RTX 5070 Ti (sm_120, Blackwell architecture)
* VRAM: 16GB
* CUDA: 12.8
* PyTorch: 2.11.0

**Model Sizes:**

* YOLOv8n: ~6M parameters (baseline)
* T5-base + LoRA: ~220M parameters (60M trainable)

**Dataset Format:**

* Images: YOLO format (640×640, normalized bboxes)
* Recipes: Parquet (structured JSON-like)

---

## Approach

**Ingredient Detection:**

* Transfer learning from YOLOv8 pretrained weights
* Focus on Indian ingredients: cumin, turmeric, paneer, ghee, garam masala
* Handle packaged vs. raw ingredient variations

**Recipe Generation:**

* T5 fine-tuned on Indian + Mexican recipe corpus
* LoRA (r=16) for efficient training
* Prompt format: `"ingredients: [list] → fusion recipe:"`
* Substitution rules: programmatic (cooking method, base material, flavor profile)

**Fusion Logic:**

* Identify base cuisines from detected ingredients
* Apply substitution taxonomy (e.g., roti ↔ tortilla based on flatbread category)
* Generate culturally coherent combinations (not arbitrary mixing)

---

## Project Structure

```
bawarchi/
├── data/
│   ├── merged/              # 9,933 training images
│   └── recipes/             # 54,857 recipe corpus
├── models/
│   ├── detection/           # YOLOv8 weights
│   └── generation/          # T5-LoRA checkpoints
├── scripts/
│   ├── train_yolo.py        # Week 2: YOLOv8 training
│   ├── train_generation.py  # Week 5: T5 training
│   └── test_model.py        # Inference testing
└── src/
    ├── detection/           # YOLOv8 inference
    ├── generation/          # T5 generation
    └── fusion/              # Substitution logic
```

---

## Known Limitations

* Missing ingredients: ghee, turmeric powder, garam masala (custom capture needed)
* No real-time video inference (static images only)

---

## Future Enhancements

* Possibly expand to more cuisines (Thai, Mediterranean)
* Add nutrition estimation
* Mobile app deployment
* Real-time webcam inference

---

## License

MIT License - see [LICENSE](https://claude.ai/chat/LICENSE)

---

## Author

**Tushar** **Jaju**

**Contact:** [GitHub](https://github.com/Tushar-9802/bawarchi)
