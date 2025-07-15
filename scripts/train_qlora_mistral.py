# scripts/train_qlora_mistral.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig, get_peft_model
)

# ─── Paths ─────────────────────────────────────────────
MODEL_DIR = "/home/pkidwell/hf_models/mistral-7b"
DATA_PATH = "/home/pkidwell/mistral_qlora/data/prompts.csv"
OUTPUT_DIR = "./qlora_mistral_output"

# ─── Load Dataset ─────────────────────────────────────
dataset = load_dataset("csv", data_files=DATA_PATH, split="train")

def format_example(example):
    return {
        "text": f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['act']}"
    }

dataset = dataset.map(format_example)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
    batched=True
)

# ─── Load Quantized Base Model ─────────────────────────
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ─── Training Setup ────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# ─── Train ─────────────────────────────────────────────
trainer.train()

# ─── Save Adapter & Tokenizer ──────────────────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

