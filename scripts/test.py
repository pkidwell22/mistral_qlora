#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1) Specify the base model repository (this will pull everything from HF)
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# 2) Load both the tokenizer and full model
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto"
)

# 3) Save them into your local “final” folder
OUTPUT_DIR = "qlora_mistral_output/final"
tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)

print(f"✅ All files saved to {OUTPUT_DIR}")
