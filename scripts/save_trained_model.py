# scripts/save_trained_model.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─── Paths ─────────────────────────────────────────────
BASE_MODEL_DIR = "/home/pkidwell/hf_models/mistral-7b"
QLORA_DIR = "./qlora_mistral_output"
FINAL_SAVE_DIR = "./qlora_mistral_output/final"

# ─── Load base model (full precision) ──────────────────
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    device_map="auto",
    trust_remote_code=True
)

# ─── Load LoRA adapter and merge ───────────────────────
model = PeftModel.from_pretrained(base_model, QLORA_DIR)
model = model.merge_and_unload()

# ─── Save merged model and tokenizer ───────────────────
model.save_pretrained(FINAL_SAVE_DIR)
print(f"✅ Model saved to {FINAL_SAVE_DIR}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
tokenizer.save_pretrained(FINAL_SAVE_DIR)
print(f"✅ Tokenizer saved to {FINAL_SAVE_DIR}")
