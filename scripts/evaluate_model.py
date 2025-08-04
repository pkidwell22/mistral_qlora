# scripts/evaluate_model.py

import os
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import PeftModel
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling

base_model_path = "C:/Users/pkidw/PycharmProjects/hf_models/mistral-7b"
adapter_path = "qlora_outputs/base_gutenberg_2ep"
eval_dataset_path = "local_datasets/gutenberg_base_1B"

# Load base model
tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", local_files_only=True)
model = PeftModel.from_pretrained(model, adapter_path)

# Load and tokenize evaluation set
dataset = load_from_disk(eval_dataset_path)
split = dataset.train_test_split(test_size=0.05, seed=42)
eval_data = split["test"]

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"]
    }

eval_data = eval_data.map(tokenize, batched=True, remove_columns=eval_data.column_names)

args = TrainingArguments(
    output_dir="eval_output",
    per_device_eval_batch_size=1,
    dataloader_drop_last=False,
)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

metrics = trainer.evaluate()
loss = metrics.get("eval_loss", None)
perplexity = math.exp(loss) if loss else float("inf")
print(f"\n📉 Eval loss: {loss:.4f}  |  🤖 Perplexity: {perplexity:.2f}")
