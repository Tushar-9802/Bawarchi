

---
---
language: en
license: mit
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
  - recipe-generation
  - llama
  - peft
  - lora
  - yolo
  - computer-vision
  - nlp
  - cuisine-aware
---


---
---

# Bawarchi Recipe Generation Model

**Role-aware fine-tuned Llama 3.2 3B for cuisine-context recipe generation**

## Model Description

Bawarchi is a recipe generation model that understands ingredient roles and adapts recipes based on cuisine context. Fine-tuned from Llama 3.2 3B using LoRA with a novel curriculum learning approach over 335K recipes, plus role-aware extension training on 1K curated examples.

**Key Capabilities:**

- **Cuisine-context switching:** Same ingredients → different dishes by cuisine (e.g., chicken + tortilla = tacos in Mexican, roti wrap in Indian)
- **Ingredient role reasoning:** Understands ingredients serve interchangeable roles (chicken ↔ paneer as protein, pasta ↔ rice as carb)
- **Intelligent adaptation:** Suggests culturally appropriate substitutions

**Example:**

```
Input: Create Indian recipe using chicken, tomato, tortilla
Output: "Chicken Tikka Roti Wrap - Note: Tortilla detected. For authentic Indian cuisine, 
consider using roti or naan instead as the flatbread base..."
```

- **Developed by:** Tushar Jaju
- **Model type:** Causal Language Model (Recipe Generation)
- **Language:** English
- **License:** MIT
- **Base Model:** meta-llama/Llama-3.2-3B-Instruct
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)

## Model Details

### Architecture

- **Base:** Llama 3.2 3B (3 billion parameters)
- **Fine-tuning:** LoRA adapters (trainable parameters: ~37M)
- **LoRA Configuration:**
  - Phase 1-3: r=64, alpha=128, dropout=0.05
  - Role-aware: r=16, alpha=32, dropout=0.05
  - Target modules: `[q_proj, k_proj, v_proj, o_proj]`
- **Precision:** fp16 (mixed precision training)
- **Context Length:** 2048 tokens

### Training Strategy

**Curriculum Learning (Phase 1-3):**

1. **Phase 1 - Simple Recipes:** Basic structure (ingredients → steps)

   - 100K examples, 3 epochs
   - Learning rate: 2e-4
2. **Phase 2 - Complex Recipes:** Multi-step reasoning, timing, techniques

   - 185K examples, 3 epochs
   - Learning rate: 2e-4
3. **Phase 3 - Fusion Recipes:** Cultural adaptation, substitution logic

   - 50K examples, 5 epochs
   - Learning rate: 1e-4

**Role-Aware Extension:**

- 1K curated examples demonstrating ingredient role reasoning
- 2 epochs, learning rate: 1e-4
- Teaches cuisine-context switching

**Total Training Time:** 13 hours (11h Phase 1-3 + 2h role-aware)

### Performance

| Metric     | Value | Description                     |
| ---------- | ----- | ------------------------------- |
| Accuracy   | 74.1% | Recipe structure + coherence    |
| Final Loss | 0.997 | Cross-entropy on validation set |
| BLEU Score | ~0.45 | N-gram overlap with references  |
| Perplexity | ~2.71 | Model confidence                |

**Cuisine-Context Validation:**

- Roti + chicken → Indian-style preparation (not Mexican)
- Tortilla + paneer → Mexican-style usage (not Indian)
- Pasta in Indian request → Suggests rice/roti substitution
- Ingredient role understanding across cultures

## Intended Use

### Primary Use Case

Generate cuisine-aware recipes from a list of available ingredients with intelligent adaptation based on cultural context.

**Supported Cuisines:**

- Indian
- Mexican
- Italian
- Asian (general)
- Fusion
- General/International

**Input Format:**

```
Create a {difficulty}-level {cuisine} recipe using: {ingredient_list}
```

**Output Format:**

- Recipe title
- Ingredient list with quantities
- Step-by-step instructions
- Cooking time and servings
- (Optional) Substitution suggestions

### Downstream Applications

- Recipe recommendation systems
- Meal planning applications
- Cooking assistants
- Dietary adaptation tools
- Culinary education platforms

### Out-of-Scope Use

**Not suitable for:**

- Medical/dietary advice without professional validation
- Nutrition calculation (no nutritional analysis capabilities)
- Allergen detection (model may miss allergens)
- Commercial recipe databases (license implications)
- Real-time video-based cooking guidance

## How to Use

### Installation

```bash
pip install transformers peft torch
```

### Basic Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
adapter_path = "Tushar-9802/bawarchi-recipe-generation"  # Replace with actual path
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# Generate recipe
ingredients = "chicken, tomato, onion, garlic, ginger"
cuisine = "indian"
difficulty = "medium"

prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Create a {difficulty}-level {cuisine} recipe using: {ingredients}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(recipe.split("assistant")[-1].strip())
```

### Advanced: Cuisine-Aware Generation

```python
def generate_cuisine_aware_recipe(ingredients, cuisine="general", difficulty="medium"):
    """
    Generate recipe with cuisine-context awareness
  
    Args:
        ingredients: List of ingredients or comma-separated string
        cuisine: Target cuisine (indian/mexican/italian/asian/fusion/general)
        difficulty: Recipe complexity (easy/medium/hard)
  
    Returns:
        Generated recipe text
    """
    if isinstance(ingredients, list):
        ingredients = ", ".join(ingredients)
  
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Create a {difficulty}-level {cuisine} recipe using: {ingredients}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
  
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
  
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
  
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("assistant")[-1].strip()

# Example usage
recipe = generate_cuisine_aware_recipe(
    ingredients=["chicken", "tomato", "tortilla", "cheese"],
    cuisine="mexican",
    difficulty="easy"
)
print(recipe)
```

## Training Details

### Training Data

**Primary Dataset:** [Food.com Recipes](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)

- 335,000 recipes (filtered and cleaned)
- Distribution: General (89.7%), Indian (6.5%), Mexican (10.2%)

**Synthetic Data:**

- 43,000 substitution-aware examples (quality filtered)
- Generated using GPT-based augmentation

**Role-Aware Extension:**

- 1,000 curated examples demonstrating:
  - Ingredient role classification
  - Cuisine-specific substitutions
  - Cultural context reasoning

**Data Processing:**

- Deduplication based on title + ingredient similarity
- Length filtering (50-2000 tokens)
- Quality filtering (coherence, completeness)
- Response template masking (only instructions in loss)

### Training Procedure

**Hardware:**

- GPU: NVIDIA RTX 5070 Ti (16GB VRAM)
- CUDA: 12.8
- Driver: 566.36

**Software:**

- PyTorch: 2.1.0
- Transformers: 4.45.0
- PEFT: 0.7.0
- Python: 3.11.7

**Hyperparameters:**

**Phase 1-3:**

```yaml
learning_rate: 2e-4 (Phase 1-2), 1e-4 (Phase 3)
batch_size: 4
gradient_accumulation_steps: 8
effective_batch_size: 32
max_length: 2048
num_epochs: 3 (Phase 1-2), 5 (Phase 3)
optimizer: AdamW
lr_scheduler: linear with warmup
warmup_steps: 100
weight_decay: 0.01
fp16: True
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
```

**Role-Aware Extension:**

```yaml
learning_rate: 1e-4
batch_size: 2
gradient_accumulation_steps: 8
effective_batch_size: 16
num_epochs: 2
lora_r: 16
lora_alpha: 32
```

**Training Time:**

- Phase 1: ~3.5 hours
- Phase 2: ~5 hours
- Phase 3: ~2.5 hours
- Role-aware: ~2 hours
- **Total: ~13 hours**

**Optimization Techniques:**

- Gradient checkpointing
- Mixed precision training (fp16)
- Response-only loss masking
- Early stopping (patience=2)
- Data loading parallelism (8 workers)

## Evaluation

### Test Set Performance

**Metrics on 1000-example test set:**

- Accuracy (structure + coherence): 74.1%
- Loss: 0.997
- BLEU-4: ~0.45
- Perplexity: ~2.71

**Qualitative Evaluation:**

- Recipe completeness: 92% (has all sections)
- Ingredient usage: 88% (uses provided ingredients)
- Cuisine appropriateness: 78% (matches selected cuisine)
- Instruction clarity: 85% (step-by-step clear)

### Limitations

**Technical:**

- Context limited to 2048 tokens (very long recipes may truncate)
- English language only
- Trained primarily on Western and South Asian cuisines
- May hallucinate rare ingredients or techniques

**Data Bias:**

- Over-representation of general/international cuisine (89.7%)
- Under-representation of regional/specialty cuisines
- Bias toward common ingredients (chicken, tomato, onion)

**Safety:**

- Does not validate food safety (temperatures, allergens)
- May suggest inappropriate ingredient combinations
- No nutritional information or dietary restriction handling

## Bias and Ethics

**Known Biases:**

- Cultural: Western/Indian recipes over-represented
- Ingredient: Common ingredients favored over specialty items
- Complexity: Bias toward medium-complexity recipes

**Ethical Considerations:**

- Recipe sources (Food.com) licensed under CC-BY-SA
- Model outputs should be validated for food safety
- Dietary restrictions require professional verification
- Allergen detection not reliable - always verify

**Recommendations:**

- Validate recipes with culinary experts for safety
- Cross-check allergen information independently
- Consider cultural sensitivity when adapting traditional recipes
- Use as a creative tool, not authoritative source

**ompute Efficiency:**

- Model size: 3B parameters (efficient for performance)
- LoRA fine-tuning: 37M trainable (vs. 3B full fine-tuning)
- Mixed precision: 2x memory efficiency
- Curriculum learning: Converged in 13h (vs. naive ~24h)

## Citation

**BibTeX:**

```bibtex
@misc{bawarchi2026,
  author = {Jaju, Tushar},
  title = {Bawarchi: Role-Aware Recipe Generation with Llama 3.2},
  year = {2026},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/Tushar-9802/bawarchi-recipe-generation}},
  note = {Fine-tuned Llama 3.2 3B with curriculum learning and role-based reasoning}
}
```

**APA:**

```
Jaju, T. (2026). Bawarchi: Role-Aware Recipe Generation with Llama 3.2 [Computer software]. 
HuggingFace. https://huggingface.co/Tushar-9802/bawarchi-recipe-generation
```

## Model Card Authors

**Tushar Jaju**

- GitHub: [@Tushar-9802](https://github.com/Tushar-9802)
- LinkedIn: [Tushar Jaju](https://linkedin.com/in/tushar-jaju)
- Email: tusharjaju98@gmail.com

## Additional Information

**Related Resources:**

- **GitHub Repository:** [Tushar-9802/bawarchi](https://github.com/Tushar-9802/bawarchi)
- **Detection Model:** YOLOv8m (66.51% mAP, 124 ingredient classes)
- **Substitution System:** PMI + Embeddings (85% precision)
- **Web Demo:** Streamlit application (see GitHub repo)
- **Dataset:** [Food.com on Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)

**Future Work:**

- Extend to more cuisines (Thai, Chinese, Mediterranean, Japanese)
- Add nutrition estimation capabilities
- Dietary restriction filtering (vegan, gluten-free, keto)
- Multi-language support (Hindi, Spanish, French)
- Larger model fine-tuning (7B, 13B parameters)

**Acknowledgments:**

- Base model: Meta AI ([Llama 3.2](https://ai.meta.com/llama/))
- Dataset: Food.com via Kaggle
- Framework: Hugging Face Transformers, PEFT
- Community: Hugging Face, PyTorch

---

**License:** MIT License

**Contact:** For questions, issues, or collaboration opportunities, please open an issue on [GitHub](https://github.com/Tushar-9802/bawarchi) or contact via email.

### Framework Versions

- PEFT: 0.7.0
- Transformers: 4.45.0
- PyTorch: 2.1.0
- Python: 3.11.7
