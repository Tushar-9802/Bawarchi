---
library_name: peft
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
- recipe-generation
- fusion-cuisine
- llama-3.2
- peft
- lora
license: mit
---

# Bawarchi Recipe Generation - Final Adapter

LoRA adapter for Llama 3.2 3B-Instruct, fine-tuned for fusion recipe generation from ingredients.

## Performance

- **Loss:** 0.997
- **Token Accuracy:** 74.1%
- **Training:** 11 hours, 3-phase curriculum

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = PeftModel.from_pretrained(model, "models/bawarchi-adapter/final")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
```

## Training Details

- **Data:** 335K recipes (292K base + 43K synthetic)
- **LoRA:** rank=64, alpha=16
- **Batch:** 2 (effective 16)
- **Hardware:** RTX 5070 Ti

Repository: https://github.com/Tushar-9802/bawarchi
