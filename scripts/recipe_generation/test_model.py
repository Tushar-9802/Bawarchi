"""
Test trained Llama 3.2 3B recipe generation model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path


def load_model(adapter_path="models/bawarchi-adapter/final"):
    """Load base model + trained adapter"""
    print("Loading model...")
    
    base_model = "meta-llama/Llama-3.2-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Loaded from {adapter_path}")
    return model, tokenizer


def generate_recipe(model, tokenizer, ingredients, cuisine="indian", difficulty="medium"):
    """Generate recipe"""
    
    ingredients_str = ", ".join(ingredients)
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Create a {difficulty}-level {cuisine} recipe using: {ingredients_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"\nIngredients: {ingredients_str}")
    print(f"Type: {cuisine} ({difficulty})")
    print("-" * 70)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    recipe = response.split("assistant")[-1].strip() if "assistant" in response else response
    
    return recipe


def main():
    print("=" * 70)
    print("BAWARCHI - RECIPE GENERATION TEST")
    print("=" * 70)
    
    adapter_path = Path("models/bawarchi-adapter/final")
    if not adapter_path.exists():
        print(f"✗ Model not found: {adapter_path}")
        return
    
    model, tokenizer = load_model(str(adapter_path))
    
    # Test 1
    print("\n" + "=" * 70)
    print("TEST 1: INDIAN RECIPE")
    print("=" * 70)
    recipe = generate_recipe(
        model, tokenizer,
        ["chicken", "tomato", "onion", "garlic", "ginger"],
        "indian", "medium"
    )
    print(recipe)
    
    # Test 2
    print("\n" + "=" * 70)
    print("TEST 2: FUSION RECIPE")
    print("=" * 70)
    recipe = generate_recipe(
        model, tokenizer,
        ["pasta", "paneer", "curry leaves", "coconut milk"],
        "fusion", "medium"
    )
    print(recipe)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()