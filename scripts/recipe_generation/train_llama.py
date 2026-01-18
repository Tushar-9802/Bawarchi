"""
Llama 3.2 3B Recipe Generation Training - OPTIMIZED
Compatible with transformers 4.45+ / TRL 0.26+
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    max_seq_length = 2048
    
    # LoRA
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # Training
    batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    
    # Eval optimization
    max_eval_samples = 2000
    
    # Paths
    data_dir = Path("data/training")
    output_dir = Path("models/bawarchi-adapter")
    logs_dir = Path("results/recipe_generation")
    
    # Curriculum
    curriculum_phases = [
        {"name": "simple", "epoch": 1, "max_ingredients": 6},
        {"name": "complex", "epoch": 2, "max_ingredients": 12},
        {"name": "fusion", "epoch": 3, "max_ingredients": 15}
    ]

# ============================================================================
# LORA
# ============================================================================

def setup_lora():
    return LoraConfig(
        r=TrainingConfig.lora_r,
        lora_alpha=TrainingConfig.lora_alpha,
        target_modules=TrainingConfig.lora_target_modules,
        lora_dropout=TrainingConfig.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

# ============================================================================
# MODEL
# ============================================================================

def load_model_and_tokenizer():
    print("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        TrainingConfig.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = TrainingConfig.max_seq_length
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        TrainingConfig.model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_cache=False,
    )
    
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, setup_lora())
    
    print("Model loaded. Trainable parameters:")
    model.print_trainable_parameters()
    
    return model, tokenizer

# ============================================================================
# DATA
# ============================================================================

def format_instruction(example):
    system_prompts = [
        "You are Bawarchi, an expert chef AI that generates detailed, context-aware recipes.",
        "You are an expert culinary AI that creates practical, well-structured recipes.",
        "You are a professional chef AI specializing in recipe generation.",
        "You are Bawarchi, a culinary expert that generates recipes with deep understanding of ingredients.",
    ]
    
    system_idx = hash(example.get('instruction', '')) % len(system_prompts)
    system_prompt = system_prompts[system_idx]
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
    
    return {"text": prompt}


def load_training_data(phase=None):
    print("Loading training data...")
    
    train_file = TrainingConfig.data_dir / "train_instructions.jsonl"
    val_file = TrainingConfig.data_dir / "val_instructions.jsonl"
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation data not found: {val_file}")
    
    train_data = load_dataset('json', data_files=str(train_file), split='train')
    val_data = load_dataset('json', data_files=str(val_file), split='train')
    
    if phase:
        max_ing = phase['max_ingredients']
        train_data = train_data.filter(lambda x: len(x.get('ingredients', [])) <= max_ing)
        print(f"Phase: {phase['name']}, max_ingredients: {max_ing}")
        print(f"Training examples: {len(train_data)}")
    
    # Subset validation for speed
    if len(val_data) > TrainingConfig.max_eval_samples:
        val_data = val_data.shuffle(seed=42).select(range(TrainingConfig.max_eval_samples))
        print(f"Validation subset: {len(val_data)}")
    else:
        print(f"Validation: {len(val_data)}")
    
    train_data = train_data.map(format_instruction, remove_columns=train_data.column_names)
    val_data = val_data.map(format_instruction, remove_columns=val_data.column_names)
    
    return train_data, val_data

# ============================================================================
# TRAINER (MINIMAL VALIDATED ARGS)
# ============================================================================

def setup_trainer(model, tokenizer, train_data, val_data, phase_num):
    output_dir = str(TrainingConfig.output_dir / f"phase_{phase_num}")
    logging_dir = str(TrainingConfig.logs_dir / f"phase_{phase_num}")
    
    # MINIMAL TrainingArguments - only guaranteed-compatible args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        
        # Eval - use eval_strategy (newer) with fallback
        eval_strategy="steps",
        eval_steps=1000,
        
        # Optimizer
        learning_rate=TrainingConfig.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Memory
        gradient_checkpointing=True,
        fp16=True,
        
        # Logging
        logging_dir=logging_dir,
        logging_steps=100,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=False,
        
        # Dataloader
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        
        # Misc
        seed=42,
        report_to="tensorboard",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    
    return trainer

# ============================================================================
# TRAINING
# ============================================================================

def train_curriculum():
    model, tokenizer = load_model_and_tokenizer()
    TrainingConfig.output_dir.mkdir(parents=True, exist_ok=True)
    TrainingConfig.logs_dir.mkdir(parents=True, exist_ok=True)
    
    best_eval_loss = float('inf')
    best_phase = None
    
    for i, phase in enumerate(TrainingConfig.curriculum_phases, 1):
        print(f"\n{'='*70}")
        print(f"PHASE {i}/3: {phase['name'].upper()}")
        print(f"{'='*70}\n")
        
        train_data, val_data = load_training_data(phase)
        trainer = setup_trainer(model, tokenizer, train_data, val_data, i)
        
        # Resume logic
        resume_from = None
        if i > 1:
            prev_dir = TrainingConfig.output_dir / f"phase_{i-1}"
            checkpoints = list(prev_dir.glob("checkpoint-*"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: int(p.name.split('-')[1]))
                resume_from = str(latest)
                print(f"Resuming from: {resume_from}")
        
        print(f"Training phase {i}...")
        trainer.train(resume_from_checkpoint=resume_from)
        
        print(f"Evaluating phase {i}...")
        eval_results = trainer.evaluate()
        
        if eval_results['eval_loss'] < best_eval_loss:
            best_eval_loss = eval_results['eval_loss']
            best_phase = i
            print(f"✓ New best! Loss: {best_eval_loss:.4f}")
        
        # Save results
        with open(TrainingConfig.logs_dir / f"phase_{i}_results.json", 'w') as f:
            json.dump({**eval_results, 'phase': i, 'is_best': (i == best_phase)}, f, indent=2)
        
        print(f"Phase {i} done. Loss: {eval_results['eval_loss']:.4f}")
        
        # Save phase
        phase_path = TrainingConfig.output_dir / f"phase_{i}_final"
        trainer.save_model(str(phase_path))
        tokenizer.save_pretrained(str(phase_path))
        
        model = trainer.model
        torch.cuda.empty_cache()
    
    # Save final
    print(f"\nBest: phase {best_phase}, loss {best_eval_loss:.4f}")
    
    final_path = TrainingConfig.output_dir / "final"
    best_path = TrainingConfig.output_dir / f"phase_{best_phase}_final"
    
    import shutil
    if final_path.exists():
        shutil.rmtree(final_path)
    shutil.copytree(best_path, final_path)
    
    print(f"Final adapter: {final_path}")
    
    with open(TrainingConfig.logs_dir / "summary.json", 'w') as f:
        json.dump({
            "best_phase": best_phase,
            "best_loss": float(best_eval_loss),
            "path": str(final_path)
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("BAWARCHI - OPTIMIZED TRAINING")
    print("="*70)
    print(f"\nOptimizations: eval_steps=1000, eval_samples=2000, load_best=False")
    print(f"Expected: ~7h/phase, ~21h total\n")
    
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    
    if not torch.cuda.is_available():
        print("ERROR: No CUDA")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    train_curriculum()

if __name__ == "__main__":
    main()