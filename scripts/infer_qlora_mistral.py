# scripts/infer_qlora_mistral.py

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

# ─── Config ─────────────────────────────────────────────
MODEL_DIR = "/home/pkidwell/mistral_qlora/qlora_mistral_output/final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load Model and Tokenizer ──────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", torch_dtype=torch.float16)

# ─── Inference ─────────────────────────────────────────
prompt = "### Instruction:\nExplain quantum computing like I'm five.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

with torch.no_grad():
    _ = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
