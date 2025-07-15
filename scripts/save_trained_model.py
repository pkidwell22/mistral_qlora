# scripts/save_trained_model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ─── Paths ─────────────────────────────────────────────
BASE_MODEL_DIR = "/home/pkidwell/hf_models/mistral-7b"
QLORA_DIR = "./qlora_mistral_output"
FINAL_SAVE_DIR = "./qlora_mistral_output/final"

# ─── Load base model in 4-bit ──────────────────────────
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# ─── Load fine-tuned LoRA weights ──────────────────────
model = PeftModel.from_pretrained(base_model, QLORA_DIR)
model = model.merge_and_unload()

# ─── Save full model ───────────────────────────────────
model.save_pretrained(FINAL_SAVE_DIR)
print(f"✅ Model saved to {FINAL_SAVE_DIR}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
tokenizer.save_pretrained(FINAL_SAVE_DIR)
print(f"✅ Tokenizer saved to {FINAL_SAVE_DIR}")
