# scripts/chat_gradio.py

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

# ─── Config ─────────────────────────────────────────────────────────────
BASE_MODEL = "/mnt/c/Users/pkidw/hf_models/mistral-7b"
ADAPTER_PATH = "qlora_hle_output"

# ─── Load Tokenizer ─────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

# ─── Load Model with LoRA ───────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# ─── Chat Function ──────────────────────────────────────────────────────
def chat(message, history):
    prompt = f"### Instruction:\n{message}\n\n### Response:\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

# ─── Launch Gradio Interface ─────────────────────────────────────────────
gr.ChatInterface(fn=chat, title="🧠 Mistral Chat (HLE LoRA)").launch()
