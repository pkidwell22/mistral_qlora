import os
import argparse
from pathlib import Path
from datasets import load_from_disk, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

# ─── Args ───────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--tokenizer_path", type=str, required=True)
parser.add_argument("--dataset_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--grad_accum", type=int, default=8)
parser.add_argument("--wandb_project", type=str, default="qlora-gutenberg")
args = parser.parse_args()

# ─── Init Weights & Biases ─────────────────────────────
os.environ["WANDB_PROJECT"] = args.wandb_project
wandb.init(project=args.wandb_project, name="qlora-gutenberg-full")

# ─── Load Tokenizer ────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    Path(args.tokenizer_path).as_posix(),
    local_files_only=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─── Load Full Dataset ─────────────────────────────────
chunk_paths = sorted(Path(args.dataset_dir).glob("chunk_*"))
print(f"\U0001F4DA Loading {len(chunk_paths)} chunks...")
datasets = [load_from_disk(p.as_posix()) for p in chunk_paths]
full_dataset = concatenate_datasets(datasets)
dataset = DatasetDict({"train": full_dataset})
print(f"✅ Dataset loaded: {len(dataset['train'])} train examples")

# ─── LoRA Model Setup ─────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)
model = AutoModelForCausalLM.from_pretrained(
    Path(args.model_path).as_posix(),
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True
)
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=128,
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ─── Format Labels ─────────────────────────────────────
def format_for_causal_lm(example):
    example["labels"] = example["input_ids"]
    return example
dataset = dataset.map(format_for_causal_lm, batched=False)

# ─── Trainer Setup ─────────────────────────────────────
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    bf16=True,
    optim="paged_adamw_32bit",
    report_to="wandb",
    run_name="qlora-gutenberg-full",
    logging_dir=os.path.join(args.output_dir, "logs")
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

# ─── Train ─────────────────────────────────────────────
print("\U0001F680 Starting training...")
trainer.train()

# ─── Save Adapter Only ────────────────────────────────
print("\U0001F4BE Saving LoRA adapter...")
model.save_pretrained(args.output_dir)
print("✅ Done.")
