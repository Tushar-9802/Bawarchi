# Bawarchi

**AI-powered ingredient detection and role-aware recipe generation**

Complete ML system combining computer vision (YOLOv8m) and natural language processing (Llama 3.2 3B) to detect ingredients from images and generate cuisine-aware recipes with intelligent substitution suggestions.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What It Does

**Input:** Upload a photo of ingredients (or enter manually)

**Output:** Cuisine-aware recipe that adapts to available ingredients

**Key Innovation:** Role-based ingredient reasoning - understands that chicken and paneer serve the same role (protein), so the same ingredients can become different dishes based on cuisine selection.

**Example:**

```
Ingredients: Chicken, tomato, tortilla
→ Mexican cuisine: Chicken Tacos (uses tortilla)
→ Indian cuisine: Chicken Roti Wrap (suggests roti instead)
```

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Detection  │────>│ Substitution │────>│  Generation  │
│  YOLOv8m    │     │  PMI + Emb   │     │  Llama 3.2B  │
│  66.51%     │     │     85%      │     │    74.1%     │
└─────────────┘     └──────────────┘     └──────────────┘
                            │
                            │
        ┌───────────────────┴──────────────────┐
        │         Streamlit Web Interface       │
        │  Detection → Preparation → Recipe     │
        └───────────────────────────────────────┘
```

---

## Project Status: v1.0 Complete

**All core phases completed (8-week development):**

* Phase 1: Dataset acquisition and validation (9,933 images, 380K recipes)
* Phase 2: YOLOv8m detection model training (66.51% mAP)
* Phase 3: Substitution learning system (PMI + embeddings, 85% precision)
* Phase 4: Recipe generation training (335K recipes, 11 hours)
* Phase 5: Role-aware fine-tuning (1K examples, 2 hours)
* Phase 6: End-to-end Streamlit UI integration
* Phase 7: Cuisine-context switching validation
* Phase 8: Production deployment preparation

**Deployment Ready:** GitHub ✓ | HuggingFace (models) | Kaggle (datasets)

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Tushar-9802/bawarchi.git
cd bawarchi

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Run Application

```bash
# Launch Streamlit interface
streamlit run app.py

# Access at: http://localhost:8501
```

### Usage

1. **Preparation Tab:**
   * Upload image / Use camera / Manual entry
   * Select detected ingredients
   * (Optional) View ingredient substitution suggestions
2. **Recipe Generation Tab:**
   * Select cuisine (Indian/Mexican/Italian/Asian/Fusion/General)
   * Select difficulty level
   * Generate recipe
   * View cuisine-adapted recipe

---

## Model Performance

### Detection Model (YOLOv8m)

| Metric          | Value   | Notes                        |
| --------------- | ------- | ---------------------------- |
| mAP@0.5         | 66.51%  | 6.51% above target           |
| mAP@0.5:0.95    | 41.10%  | COCO-style metric            |
| Precision       | 68.91%  | Low false positive rate      |
| Recall          | 59.14%  | Balanced for real-world use  |
| Parameters      | 25.9M   | YOLOv8m architecture         |
| Inference Speed | ~35 FPS | RTX 5070 Ti                  |
| Classes         | 124     | Food ingredients             |
| Training Time   | 4 hours | 100 epochs (converged at 97) |

**Training Hardware:** RTX 5070 Ti (16GB VRAM)

### Substitution System

| Metric           | Value  | Implementation          |
| ---------------- | ------ | ----------------------- |
| Precision        | 85.15% | PMI + embeddings (RRF)  |
| Ingredient Pairs | 15,376 | From 522K recipe corpus |
| Embedding Dim    | 384    | sentence-transformers   |
| Category Filter  | ✓     | +6% precision boost     |

**Key Features:**

* Automated learning from recipe co-occurrence
* Semantic similarity via sentence embeddings
* PMI-based statistical relationships
* Category-aware filtering (protein↔protein, grain↔grain)

### Recipe Generation (Llama 3.2 3B)

| Metric           | Value    | Configuration                 |
| ---------------- | -------- | ----------------------------- |
| Accuracy         | 74.1%    | Recipe structure + coherence  |
| Loss (final)     | 0.997    | Cross-entropy                 |
| Training Time    | 11 hours | Phase 1-3 curriculum learning |
| Fine-tuning Time | 2 hours  | Role-aware extension          |
| Training Data    | 335K     | Recipes + 1K role examples    |
| LoRA Rank        | 64       | Phase 1-3 / 16 for role-aware |
| Max Length       | 2048     | Input + output tokens         |
| Batch Size       | 4        | With gradient accumulation=8  |

**Training Strategy:**

* **Phase 1:** Simple recipes (100K examples, 3 epochs)
* **Phase 2:** Complex multi-step recipes (185K examples, 3 epochs)
* **Phase 3:** Fusion recipes with substitutions (50K examples, 5 epochs)
* **Role-Aware Extension:** Ingredient role reasoning (1K examples, 2 epochs)

**Cuisine-Context Validation:**

* ✓ Roti + chicken → Indian-style wrap (not Mexican taco)
* ✓ Tortilla + paneer → Mexican-style quesadilla (not Indian paratha)
* ✓ Pasta detected in Indian request → Suggests rice/roti substitution
* ✓ Role-based reasoning: Understands ingredient interchangeability

---

## Key Features

### 1. Cuisine-Context Switching

**Problem:** Traditional recipe systems ignore cultural context.

**Solution:** Role-aware fine-tuning teaches the model that:

* Ingredients serve roles (protein, carb, aromatic, spice)
* Roles are interchangeable within cuisine context
* Same ingredients → different dishes by cuisine

**Example:**

```
Input: chicken, tomato, tortilla
Cuisine: Mexican → Chicken Tacos (uses tortilla)
Cuisine: Indian → Chicken Roti Wrap (suggests roti instead)
```

### 2. Intelligent Substitution Learning

**Problem:** Manual ingredient taxonomies are incomplete and inflexible.

**Solution:** Automated learning from 522K recipe corpus:

* **PMI (Pointwise Mutual Information):** Statistical co-occurrence
* **Semantic Embeddings:** Meaning-based similarity
* **RRF (Reciprocal Rank Fusion):** Combines both approaches

**Results:** 85% precision on ingredient substitution ranking

### 3. Curriculum Learning for Quality

**Problem:** Naive training produces inconsistent recipe quality.

**Solution:** 3-phase training strategy:

1. **Simple recipes:** Learn basic structure (ingredients → steps)
2. **Complex recipes:** Multi-step reasoning, timing, techniques
3. **Fusion recipes:** Cultural adaptation, substitution logic

**Results:** 74.1% accuracy (vs. ~60% with naive training)

---

## User Interface

### Streamlit Web Application

**Two-Tab Workflow:**

**Tab 1: Preparation**

* Input methods: Image upload, camera, manual entry
* Real-time ingredient detection with confidence scores
* Ingredient selection and management
* Detection confidence slider (0.10-0.95)

**Tab 2: Recipe Generation**

* Cuisine selection (6 options)
* Difficulty level (Easy/Medium/Hard)
* One-click recipe generation
* Clean markdown rendering
* Download as text file

**Design Principles:**

* Professional green button theme
* Light mode optimized for readability
* Minimal, distraction-free interface
* Mobile-friendly responsive layout

---

## Known Limitations

### Detection

* Weak performance on under-represented classes (<10 images)
* Missing common ingredients: ghee, turmeric powder, garam masala
* Struggles with heavily processed/cooked ingredients
* Static image only (no video support)

### Recipe Generation

* Partially trained (11h vs. optimal 24h)
* Can hallucinate rare ingredients or techniques
* Limited to 2048 token context (very long recipes may truncate)
* English language only

### Substitution System

* Coverage limited to 124 ingredient classes
* UI integration incomplete (backend works, frontend needs polish)
* No user feedback loop yet

### General

* Requires GPU for reasonable inference speed
* Model weights not included in repo (download separately)
* No nutrition information or dietary restrictions

---

## Future Enhancements

**Short-term (v1.1):**

* [ ] Fix substitution UI integration
* [ ] Add nutrition estimation API
* [ ] Expand to 200+ ingredient classes
* [ ] Multi-language support (Hindi, Spanish)

**Medium-term (v2.0):**

* [ ] Video/camera real-time detection
* [ ] Voice input for ingredients
* [ ] Save favorite recipes
* [ ] User feedback loop for quality improvement
* [ ] Mobile app (React Native)

**Long-term (v3.0):**

* [ ] Expand cuisine coverage (Thai, Chinese, Mediterranean, Japanese)
* [ ] Dietary restriction filtering (vegan, gluten-free, keto)
* [ ] Meal planning and grocery lists
* [ ] Community recipe sharing
* [ ] Integration with smart kitchen devices

---

## Development Timeline

| Phase | Duration | Milestone                                        |
| ----- | -------- | ------------------------------------------------ |
| 1     | Week 1   | Dataset acquisition (9,933 images, 380K recipes) |
| 2     | Week 2   | YOLOv8 detection training (66.51% mAP)           |
| 3     | Week 3   | Substitution system (PMI + embeddings)           |
| 4     | Week 4-5 | Recipe generation training (Phase 1-3)           |
| 5     | Week 6-7 | Role-aware fine-tuning, Streamlit UI Development |
| 6     | Week 7   | Integration, testing, deployment prep            |

**Total:** 7 weeks, ~90 hours development time

---


## Pre-trained Models

Download the trained adapter from HuggingFace:

**Recipe Generation Model:**

- Repository: [Tushar9802/bawarchi-recipe-generator](https://huggingface.co/Tushar-9802/bawarchi-recipe-generator)
- Size: ~300 MB
- Type: LoRA adapter for Llama 3.2 3B

### Quick Load

```
python
from transformers import AutoModelForCausalLM
from peft import PeftModelmodel = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = PeftModel.from_pretrained(model, "Tushar-9802/bawarchi-recipe-generator")
```

---



## License

This project is licensed under the MIT License - see the [LICENSE](https://claude.ai/chat/LICENSE) file for details.

---

## Acknowledgments

**Datasets:**

* [Roboflow Food-Ingredients Dataset](https://universe.roboflow.com/)
* [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
* [Food.com Recipes (Kaggle)](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)

**Models:**

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Meta Llama 3.2](https://ai.meta.com/llama/)
* [Sentence Transformers](https://www.sbert.net/)

**Tools:**

* [Streamlit](https://streamlit.io/)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers)
* [PyTorch](https://pytorch.org/)

---

## Contact

**Tushar Jaju**

* GitHub: [@Tushar-9802](https://github.com/Tushar-9802)
* LinkedIn: [Tushar Jaju](https://linkedin.com/in/tushar-jaju)

**Project Link:** [https://github.com/Tushar-9802/bawarchi](https://github.com/Tushar-9802/bawarchi)

---

## Project Stats

![Models Trained](https://img.shields.io/badge/Models%20Trained-3-green)![Training Hours](https://img.shields.io/badge/Training%20Hours-15-orange)![Dataset Size](https://img.shields.io/badge/Dataset-380K%20recipes-red)

**Built for food lovers and ML enthusiasts**
