#!/usr/bin/env python3
"""
Train a stacked LoRA adapter on top of an existing adapter
Optimized for Python/LLM development expertise
"""

import argparse
import os
import sys
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    set_seed
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from datasets import load_dataset, load_from_disk
import wandb


class CustomCallback(TrainerCallback):
    """Custom callback for monitoring training"""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Print key metrics
            if "loss" in logs:
                print(f"Step {state.global_step}: loss={logs['loss']:.4f}")
            if "eval_loss" in logs:
                print(f"Step {state.global_step}: eval_loss={logs['eval_loss']:.4f}")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\nEpoch {state.epoch} completed!")


class StackedAdapterTrainer:
    """Trainer for stacking LoRA adapters"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set seed for reproducibility
        set_seed(args.seed)

        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    def load_base_model_with_adapter(self):
        """Load base model with existing adapter"""
        print(f"\nLoading base model: {self.args.model_path}")

        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load base model
        if self.args.model_path.startswith("./") or self.args.model_path.startswith("/"):
            # Local model
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                local_files_only=True
            )
        else:
            # HuggingFace model
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Load existing adapter if provided
        if self.args.base_adapter_path:
            print(f"Loading base adapter: {self.args.base_adapter_path}")
            model = PeftModel.from_pretrained(model, self.args.base_adapter_path)
            print("Base adapter loaded successfully")

            # Merge the base adapter (optional - keeps it frozen)
            if self.args.merge_base_adapter:
                print("Merging base adapter with model...")
                model = model.merge_and_unload()

        # Configure new LoRA adapter
        peft_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            target_modules=self.args.target_modules.split(",") if self.args.target_modules else None,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )

        # Apply new LoRA adapter
        if self.args.base_adapter_path and not self.args.merge_base_adapter:
            # Add new adapter on top
            model.add_adapter(peft_config, adapter_name="specialized")
            model.set_adapter("specialized")
            print("Added new specialized adapter on top of base adapter")
        else:
            # Apply LoRA to the model
            model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()

        # Disable cache for gradient checkpointing
        model.config.use_cache = False

        return model

    def load_tokenizer(self):
        """Load tokenizer"""
        print(f"\nLoading tokenizer...")

        if self.args.model_path.startswith("./") or self.args.model_path.startswith("/"):
            tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_path,
                local_files_only=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def load_dataset(self, tokenizer):
        """Load and prepare dataset"""
        print(f"\nLoading dataset: {self.args.dataset_path}")

        # Load dataset based on type
        if self.args.dataset_type == "hf":
            # Load HuggingFace dataset format
            if os.path.exists(self.args.dataset_path):
                dataset = load_from_disk(self.args.dataset_path)
            else:
                # Try loading from hub
                dataset = load_dataset(self.args.dataset_path)
        else:
            # Load JSONL format
            data_files = {}

            # Check for train file
            train_path = Path(self.args.dataset_path)
            if train_path.suffix == ".jsonl":
                data_files["train"] = str(train_path)
            else:
                train_file = train_path / "train.jsonl"
                if train_file.exists():
                    data_files["train"] = str(train_file)

            # Check for validation file
            if self.args.eval_dataset_path:
                data_files["validation"] = self.args.eval_dataset_path
            else:
                val_file = train_path / "validation.jsonl"
                if val_file.exists():
                    data_files["validation"] = str(val_file)

            dataset = load_dataset("json", data_files=data_files)

        print(f"Dataset loaded: {list(dataset.keys())}")
        print(f"Train size: {len(dataset['train'])}")
        if "validation" in dataset:
            print(f"Validation size: {len(dataset['validation'])}")

        # Tokenize dataset
        def tokenize_function(examples):
            # Use the 'text' field if it exists, otherwise format from components
            if "text" in examples:
                texts = examples["text"]
            else:
                texts = []
                for i in range(len(examples["instruction"])):
                    instruction = examples["instruction"][i]
                    input_text = examples.get("input", [""] * len(examples["instruction"]))[i]
                    output = examples["output"][i]

                    if input_text:
                        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                    else:
                        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                    texts.append(text)

            # Tokenize
            model_inputs = tokenizer(
                texts,
                max_length=self.args.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Labels are same as input_ids for CLM
            model_inputs["labels"] = model_inputs["input_ids"].clone()

            return model_inputs

        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset"
        )

        return tokenized_dataset

    def create_training_args(self):
        """Create training arguments"""

        # Calculate total steps
        if hasattr(self.args, 'train_dataset_size'):
            steps_per_epoch = self.args.train_dataset_size // (
                    self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
            )
            max_steps = steps_per_epoch * self.args.num_epochs
        else:
            max_steps = -1

        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            gradient_checkpointing=self.args.gradient_checkpointing,
            optim=self.args.optim,
            learning_rate=self.args.learning_rate,
            lr_scheduler_type=self.args.lr_scheduler_type,
            warmup_steps=self.args.warmup_steps,
            weight_decay=self.args.weight_decay,
            max_grad_norm=self.args.max_grad_norm,
            logging_dir=f"{self.args.output_dir}/logs",
            logging_steps=self.args.logging_steps,
            save_strategy="steps",
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            evaluation_strategy="steps" if self.args.do_eval else "no",
            eval_steps=self.args.eval_steps if self.args.do_eval else None,
            load_best_model_at_end=self.args.do_eval,
            metric_for_best_model="eval_loss" if self.args.do_eval else None,
            greater_is_better=False,
            bf16=self.args.bf16,
            fp16=self.args.fp16,
            tf32=True,
            dataloader_num_workers=self.args.dataloader_num_workers,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=["tensorboard", "wandb"] if self.args.use_wandb else ["tensorboard"],
            seed=self.args.seed,
        )

        return training_args

    def train(self):
        """Main training function"""

        # Initialize wandb if requested
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.wandb_run_name or f"stacked-adapter-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=vars(self.args)
            )

        # Load model and tokenizer
        model = self.load_base_model_with_adapter()
        tokenizer = self.load_tokenizer()

        # Load dataset
        dataset = self.load_dataset(tokenizer)

        # Store dataset size for training args
        self.args.train_dataset_size = len(dataset["train"])

        # Create training arguments
        training_args = self.create_training_args()

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[CustomCallback(None)]
        )

        # Start training
        print("\n" + "=" * 50)
        print("Starting training...")
        print("=" * 50 + "\n")

        train_result = trainer.train()

        # Save the model
        print("\nSaving final model...")
        trainer.save_model()

        # Save training results
        with open(os.path.join(self.args.output_dir, "train_results.json"), "w") as f:
            json.dump(train_result.metrics, f, indent=2)

        print("\nTraining completed successfully!")
        print(f"Model saved to: {self.args.output_dir}")

        # Final evaluation
        if self.args.do_eval and "validation" in dataset:
            print("\nRunning final evaluation...")
            eval_results = trainer.evaluate()

            with open(os.path.join(self.args.output_dir, "eval_results.json"), "w") as f:
                json.dump(eval_results, f, indent=2)

            print(f"Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")

        # Cleanup
        if self.args.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train a stacked LoRA adapter")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base model (local or HuggingFace)")
    parser.add_argument("--base_adapter_path", type=str, default=None,
                        help="Path to existing LoRA adapter to stack on top of")
    parser.add_argument("--merge_base_adapter", action="store_true",
                        help="Merge base adapter before adding new one")

    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to training dataset")
    parser.add_argument("--dataset_type", type=str, default="hf", choices=["hf", "jsonl"],
                        help="Dataset format type")
    parser.add_argument("--eval_dataset_path", type=str, default=None,
                        help="Path to evaluation dataset (if separate)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, default=None,
                        help="Comma-separated list of target modules (default: auto-detect)")

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Learning rate scheduler type")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=0.3,
                        help="Max gradient norm for clipping")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit",
                        help="Optimizer to use")

    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Save checkpoint frequency")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--eval_steps", type=int, default=200,
                        help="Evaluation frequency")
    parser.add_argument("--do_eval", action="store_true",
                        help="Run evaluation during training")

    # Hardware arguments
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true",
                        help="Use float16 precision")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of dataloader workers")

    # Experiment tracking
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for tracking")
    parser.add_argument("--wandb_project", type=str, default="stacked-adapters",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Validate arguments
    if args.fp16 and args.bf16:
        raise ValueError("Cannot use both fp16 and bf16")

    # Auto-detect target modules if not specified
    if args.target_modules is None:
        if "mistral" in args.model_path.lower() or "mixtral" in args.model_path.lower():
            args.target_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        elif "llama" in args.model_path.lower():
            args.target_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        else:
            # Default modules that work for most models
            args.target_modules = "q_proj,k_proj,v_proj,o_proj"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Create trainer and start training
    trainer = StackedAdapterTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()