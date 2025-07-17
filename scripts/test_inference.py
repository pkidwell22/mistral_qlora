# File: scripts/test_inference_hle.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model = "/mnt/c/Users/pkidw/hf_models/mistral-7b"
adapter_path = "adapters/hle_adapter"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# Load base model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)

# Sample input
prompt = "### Instruction:\nWhat are some philosophical insights from art?\n\n### Response:\n"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# Generate
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=150, do_sample=True, top_p=0.9)

print(tokenizer.decode(output[0], skip_special_tokens=True))
