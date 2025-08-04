# scripts/infer_qlora_mistral.py
#!/usr/bin/env python3
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel

# ─── Config ─────────────────────────────────────────────
BASE_MODEL_DIR = "/mnt/c/Users/pkidw/hf_models/mistral-7b"
ADAPTER_DIR = "/home/pkidwell/mistral_qlora/qlora_mistral_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load Tokenizer from BASE MODEL ────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
)

# ─── Load Base Model and Merge LoRA Adapters ───────────
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    local_files_only=True,
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
    is_trainable=False,
).to(DEVICE)

# ─── CLI Arguments ─────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate text with Mistral QLoRA")
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--repetition_penalty", type=float, default=1.1)
args = parser.parse_args()

# ─── Prompt Format ─────────────────────────────────────
template = f"### Instruction:\n{args.prompt}\n\n### Response:\n"
inputs = tokenizer(template, return_tensors="pt").to(DEVICE)

# ─── Optional Stop Sequence ────────────────────────────
stop_id = tokenizer.encode("\n###", add_special_tokens=False)[0]

class StopOnSequence(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0][-1] == stop_id

stopping_criteria = StoppingCriteriaList([StopOnSequence()])

# ─── Stream Output ─────────────────────────────────────
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# ─── Generate Text ─────────────────────────────────────
with torch.no_grad():
    _ = model.generate(
        **inputs,
        streamer=streamer,
        stopping_criteria=stopping_criteria,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=True,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )
