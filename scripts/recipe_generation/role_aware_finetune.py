"""
ROLE-AWARE FINE-TUNING (STABLE, EXTENSION-ONLY)

Adds ingredient-role reasoning WITHOUT destroying
previous recipe generation ability.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import PeftModel, LoraConfig, get_peft_model

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# IMPORTANT: load your already-trained adapter
BASE_ADAPTER_PATH = "models/bawarchi-adapter/final"
OUTPUT_ADAPTER_PATH = "models/bawarchi-adapter/role_aware"

DATA_PATH = "data/training/role_based_cuisine_examples.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def format_example(example):
    """
    Enforces role-aware conditioning WITHOUT overriding recipe ability
    """
    system = (
        "You are Bawarchi, an expert chef AI.\n"
        "You understand ingredient ROLES (protein, base, spice, binder).\n"
        "You adapt recipes based on cuisine and available ingredients.\n"
    )

    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{system}"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{example['instruction']}\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"{example['output']}"
        f"<|eot_id|>"
    )

    return {"text": prompt}


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load previous LoRA adapter (CRITICAL)
    model = PeftModel.from_pretrained(model, BASE_ADAPTER_PATH)

    # Add NEW LoRA layer for role-awareness
    lora_config = LoraConfig(
        r=16,                     # smaller = safer
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files=DATA_PATH,
        split="train"
    )

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    tokenized = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        ),
        batched=True
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    args = TrainingArguments(
        output_dir=OUTPUT_ADAPTER_PATH,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,          # LOWER than recipe training
        num_train_epochs=2,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(OUTPUT_ADAPTER_PATH)
    tokenizer.save_pretrained(OUTPUT_ADAPTER_PATH)

    print("✅ Role-aware adapter training complete")


if __name__ == "__main__":
    main()
