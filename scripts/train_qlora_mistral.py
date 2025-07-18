import os
import torch
import argparse
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments, Trainer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling,
    LlamaTokenizer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ─── Dataset Formatting Functions ────────────────────────────────────
def format_hle(example):
    return {
        "text": f"### Instruction:\n{example['question']}\n\n### Response:\n{example['answer']}\n\n### Rationale:\n{example['rationale']}"
    }

def format_dolly(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

def format_gutenberg(example):
    return {"text": example["text"]}

def format_orca(example):
    return {
        "text": f"### System:\n{example['system_prompt']}\n\n### User:\n{example['question']}\n\n### Assistant:\n{example['response']}"
    }

def tokenize(example, tokenizer, max_length):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)

# ─── Main Training Script ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Mistral 7B with QLoRA")
    parser.add_argument("--output_dir", default="qlora_mistral_output", help="Where to save adapters")
    parser.add_argument("--model_name", default="/mnt/c/Users/pkidw/hf_models/mistral-7b", help="Path to local base model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=2)
    args = parser.parse_args()

    # Base directory resolution
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # ─── Load and format datasets ─────────────────────────────────────
    hle_ds = load_from_disk(os.path.join(base_dir, "data", "hle"))["test"].map(format_hle)
    dolly_ds = load_from_disk(os.path.join(base_dir, "local_datasets", "dolly"))["train"].map(format_dolly)
    gutenberg_ds = load_from_disk(os.path.join(base_dir, "local_datasets", "gutenberg_en")).map(format_gutenberg)
    orca_ds = load_from_disk(os.path.join(base_dir, "local_datasets", "openorca"))["train"].map(format_orca)

    full_dataset = concatenate_datasets([hle_ds, dolly_ds, gutenberg_ds, orca_ds])
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    # ─── Load tokenizer ──────────────────────────────────────────────
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name,
        local_files_only=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ─── Tokenize datasets ───────────────────────────────────────────
    train_data = train_data.map(lambda e: tokenize(e, tokenizer, args.max_length), batched=True)
    val_data = val_data.map(lambda e: tokenize(e, tokenizer, args.max_length), batched=True)

    # ─── Load model in 4-bit ─────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )

    model = prepare_model_for_kbit_training(model)

    # ─── Apply LoRA Adapters ────────────────────────────────────────
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # ─── Training configuration ─────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
    )

    # ─── Train the model ────────────────────────────────────────────
    model.config.use_cache = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    # ─── Save adapters + tokenizer ──────────────────────────────────
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Saved LoRA adapter + tokenizer to: {args.output_dir}")

if __name__ == "__main__":
    main()
