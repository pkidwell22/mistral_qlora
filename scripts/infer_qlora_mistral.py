#!/usr/bin/env python3
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ─── Config ─────────────────────────────────────────────
MODEL_DIR = "/home/pkidwell/mistral_qlora/qlora_mistral_output/final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load Model and Tokenizer ──────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
).to(DEVICE)

# ─── Argument Parsing ───────────────────────────────────
parser = argparse.ArgumentParser(
    description="Generate text with the Alpaca-fine-tuned Mistral QLoRA model"
)
parser.add_argument(
    "--prompt",
    type=str,
    required=True,
    help="Instruction to feed the model."
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=200,
    help="Maximum number of new tokens to generate."
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature."
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.9,
    help="Nucleus sampling top-p value."
)
parser.add_argument(
    "--top_k",
    type=int,
    default=50,
    help="Top-K sampling: only sample from the K most likely tokens."
)
parser.add_argument(
    "--repetition_penalty",
    type=float,
    default=1.1,
    help="Repetition penalty to reduce loops."
)
args = parser.parse_args()

# ─── Prepare Prompt ────────────────────────────────────
template = f"### Instruction:\n{args.prompt}\n\n### Response:\n"
inputs = tokenizer(template, return_tensors="pt").to(DEVICE)

# ─── Prepare Stop Sequence ─────────────────────────────
stop_id = tokenizer.encode("\n###", add_special_tokens=False)[0]
class StopOnSequence(StoppingCriteria):
    def __call__(self, input_ids, scores):
        return input_ids[0][-1] == stop_id

stopping_criteria = StoppingCriteriaList([StopOnSequence()])

# ─── Streaming Inference ───────────────────────────────
streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

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
