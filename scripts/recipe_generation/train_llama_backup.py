"""
Llama 3.2 3B Recipe Generation Training
Rigorous instruction tuning for context-aware recipe generation.
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
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Configuration
class TrainingConfig:
    # Model
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
    num_epochs = 3
    batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    warmup_steps = 100
    
    # Paths
    data_dir = Path("data/training")
    output_dir = Path("models/bawarchi-adapter")
    logs_dir = Path("results/recipe_generation")
    
    # Curriculum learning
    curriculum_phases = [
        {"name": "simple", "epoch": 1, "max_ingredients": 6},
        {"name": "complex", "epoch": 2, "max_ingredients": 12},
        {"name": "fusion", "epoch": 3, "max_ingredients": 15}
    ]

def setup_lora():
    """Configure LoRA"""
    return LoraConfig(
        r=TrainingConfig.lora_r,
        lora_alpha=TrainingConfig.lora_alpha,
        target_modules=TrainingConfig.lora_target_modules,
        lora_dropout=TrainingConfig.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

def load_model_and_tokenizer():
    """Load model with LoRA (Windows/RTX 5070 Ti compatible)"""
    print("Loading model and tokenizer...")
    
    try:
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            TrainingConfig.model_name,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = TrainingConfig.max_seq_length
        tokenizer.padding_side = "right"
        
        # Model with LoRA (no quantization for Windows compatibility)
        # PyTorch's SDPA will automatically use optimized kernels when available
        model = AutoModelForCausalLM.from_pretrained(
            TrainingConfig.model_name,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,  # Updated from torch_dtype (deprecated)
            use_cache=False  # Required for gradient checkpointing
        )
        
        # Add LoRA
        model = get_peft_model(model, setup_lora())
        
        print(f"Model loaded. Trainable parameters:")
        model.print_trainable_parameters()
        
        return model, tokenizer
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection (model downloads from HuggingFace)")
        print("  2. Verify CUDA is available: python -c 'import torch; print(torch.cuda.is_available())'")
        print("  3. Check available VRAM (need ~14-15 GB)")
        raise

def format_instruction(example):
    """Format example as instruction-following prompt with system variance"""
    
    # Multiple system prompts for better generalization
    system_prompts = [
        """You are Bawarchi, an expert chef AI that generates detailed, context-aware recipes. You understand ingredient properties, cooking methods, substitution logic, and cultural cuisine patterns.""",
        
        """You are an expert culinary AI that creates practical, well-structured recipes. You understand how ingredients behave during cooking and can adapt recipes based on substitutions.""",
        
        """You are a professional chef AI specializing in recipe generation. You provide clear instructions, appropriate cooking methods, and handle ingredient substitutions intelligently.""",
        
        """You are Bawarchi, a culinary expert that generates recipes with deep understanding of ingredients, cooking techniques, and flavor combinations.""",
    ]
    
    # Select system prompt based on hash for consistency within example
    system_idx = hash(example.get('instruction', '')) % len(system_prompts)
    system_prompt = system_prompts[system_idx]
    
    # Llama 3.2 Instruct format
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
    
    return {"text": prompt}

def load_training_data(phase=None):
    """Load and format training data with optional curriculum filtering"""
    print("Loading training data...")
    
    train_file = TrainingConfig.data_dir / "train_instructions.jsonl"
    val_file = TrainingConfig.data_dir / "val_instructions.jsonl"
    
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_file}\n"
            "Run: python scripts/recipe_generation/prepare_training_data.py"
        )
    if not val_file.exists():
        raise FileNotFoundError(
            f"Validation data not found: {val_file}\n"
            "Run: python scripts/recipe_generation/prepare_training_data.py"
        )
    
    # Load datasets
    train_data = load_dataset(
        'json',
        data_files=str(train_file),
        split='train'
    )
    
    val_data = load_dataset(
        'json',
        data_files=str(val_file),
        split='train'
    )
    
    # Apply curriculum filtering if specified
    if phase:
        max_ing = phase['max_ingredients']
        train_data = train_data.filter(
            lambda x: len(x.get('ingredients', [])) <= max_ing
        )
        print(f"Curriculum phase: {phase['name']}, max ingredients: {max_ing}")
        print(f"Filtered to {len(train_data)} examples")
    
    # Format as instructions
    train_data = train_data.map(format_instruction, remove_columns=train_data.column_names)
    val_data = val_data.map(format_instruction, remove_columns=val_data.column_names)
    
    return train_data, val_data

def setup_trainer(model, tokenizer, train_data, val_data, phase_num):
    """Configure trainer with curriculum-aware settings"""
    
    training_args = TrainingArguments(
        # Output
        output_dir=str(TrainingConfig.output_dir / f"phase_{phase_num}"),
        
        # Training - REDUCED for no quantization
        num_train_epochs=1,
        per_device_train_batch_size=4,  
        gradient_accumulation_steps=4,   # Decresed from 8
        per_device_eval_batch_size=4,
        dataloader_num_workers=4,        # Parallel data loading
        dataloader_pin_memory=True,       # Faster GPU transfer
        dataloader_prefetch_factor=2,     # Prefetch batches
        # Optimization
        learning_rate=TrainingConfig.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Memory - fp16 instead of bf16 for better compatibility
        gradient_checkpointing=True,
        fp16=True,  # Use fp16 instead of bf16
        
        # Logging
        logging_dir=str(TrainingConfig.logs_dir / f"phase_{phase_num}"),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        
        # Reporting
        report_to="none",  # Changed from tensorboard since it might cause issues
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # MINIMAL SFTTrainer for TRL 0.26.2
    # The data is already formatted, just pass it
    from trl import SFTTrainer
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    
    return trainer

def train_curriculum():
    """Train with curriculum learning across 3 phases"""
    
    try:
        # Initialize
        model, tokenizer = load_model_and_tokenizer()
        TrainingConfig.output_dir.mkdir(parents=True, exist_ok=True)
        TrainingConfig.logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"\nERROR: Failed to initialize training: {e}")
        raise
    
    # Track best model across all phases
    best_eval_loss = float('inf')
    best_phase = None
    
    # Training loop
    for i, phase in enumerate(TrainingConfig.curriculum_phases, 1):
        print(f"\n{'='*70}")
        print(f"PHASE {i}/{len(TrainingConfig.curriculum_phases)}: {phase['name'].upper()}")
        print(f"{'='*70}\n")
        
        # Load data for this phase
        train_data, val_data = load_training_data(phase)
        
        # Setup trainer
        trainer = setup_trainer(model, tokenizer, train_data, val_data, i)
        
        # Train
        print(f"Training phase {i}...")
        
        # Resume from checkpoint if phase 2 or 3
        resume_from = None
        if i > 1:
            prev_phase_dir = TrainingConfig.output_dir / f"phase_{i-1}"
            checkpoints = list(prev_phase_dir.glob("checkpoint-*"))
            if checkpoints:
                # Get latest checkpoint
                latest = max(checkpoints, key=lambda p: int(p.name.split('-')[1]))
                resume_from = str(latest)
                print(f"Resuming from checkpoint: {resume_from}")
        
        try:
            trainer.train(resume_from_checkpoint=resume_from)
        except Exception as e:
            print(f"\nERROR: Training failed in phase {i}: {e}")
            print("Training checkpoint may be corrupted. Consider restarting from a previous phase.")
            raise
        
        # Evaluate
        print(f"Evaluating phase {i}...")
        try:
            eval_results = trainer.evaluate()
        except Exception as e:
            print(f"\nWARNING: Evaluation failed in phase {i}: {e}")
            print("Continuing to next phase...")
            eval_results = {'eval_loss': float('inf')}
        
        # Track best model
        if eval_results['eval_loss'] < best_eval_loss:
            best_eval_loss = eval_results['eval_loss']
            best_phase = i
            print(f"✓ New best model! Loss: {best_eval_loss:.4f}")
        
        # Save phase results
        results_file = TrainingConfig.logs_dir / f"phase_{i}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                **eval_results,
                'phase': i,
                'phase_name': phase['name'],
                'is_best': (i == best_phase)
            }, f, indent=2)
        
        print(f"Phase {i} complete. Eval loss: {eval_results['eval_loss']:.4f}")
        
        # Save phase checkpoint
        phase_save_path = TrainingConfig.output_dir / f"phase_{i}_final"
        trainer.save_model(str(phase_save_path))
        print(f"Phase {i} model saved to: {phase_save_path}")
    
    # Save final adapter (from best phase)
    print(f"\nBest model was from phase {best_phase} with loss {best_eval_loss:.4f}")
    print("Saving final adapter...")
    
    final_path = TrainingConfig.output_dir / "final"
    
    # Load and save best phase model
    best_phase_path = TrainingConfig.output_dir / f"phase_{best_phase}_final"
    if best_phase_path.exists():
        # Copy best phase to final
        import shutil
        if final_path.exists():
            shutil.rmtree(final_path)
        shutil.copytree(best_phase_path, final_path)
    else:
        # Fallback: save current model
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))
    
    print(f"Final adapter saved to: {final_path}")
    
    # Save training summary
    summary = {
        "model": TrainingConfig.model_name,
        "lora_r": TrainingConfig.lora_r,
        "lora_alpha": TrainingConfig.lora_alpha,
        "num_phases": len(TrainingConfig.curriculum_phases),
        "total_epochs": len(TrainingConfig.curriculum_phases),
        "best_phase": best_phase,
        "best_eval_loss": float(best_eval_loss),
        "final_adapter_path": str(final_path),
        "phase_results": [
            f"phase_{i}: {(TrainingConfig.logs_dir / f'phase_{i}_results.json')}"
            for i in range(1, len(TrainingConfig.curriculum_phases) + 1)
        ]
    }
    
    with open(TrainingConfig.logs_dir / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best model: Phase {best_phase}")
    print(f"Best eval loss: {best_eval_loss:.4f}")
    print(f"Final adapter: {final_path}")

def main():
    print("="*70)
    print("BAWARCHI - LLAMA 3.2 3B RECIPE GENERATION TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {TrainingConfig.model_name}")
    print(f"  LoRA rank: {TrainingConfig.lora_r}")
    print(f"  Batch size: {TrainingConfig.batch_size}")
    print(f"  Gradient accumulation: {TrainingConfig.gradient_accumulation_steps}")
    print(f"  Effective batch size: {TrainingConfig.batch_size * TrainingConfig.gradient_accumulation_steps}")
    print(f"  Learning rate: {TrainingConfig.learning_rate}")
    print(f"  Curriculum phases: {len(TrainingConfig.curriculum_phases)}")
    print(f"\nEstimated VRAM usage: ~14-15 GB / 16 GB (without quantization)")
    print(f"Estimated training time: 5-6 hours on RTX 5070 Ti")
    print(f"\n{'='*70}\n")
    
    # Set deterministic training seeds for reproducibility
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set to False for debugging, True for speed
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"Random seed set to: {seed}")
    print(f"Deterministic mode: {'ON (slower)' if torch.backends.cudnn.deterministic else 'OFF (faster)'}\n")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. GPU required for training.")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    # Train
    train_curriculum()

if __name__ == "__main__":
    main()