#!/usr/bin/env python3
import os
# force offline mode
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import torch
import sacrebleu
from rouge_score import rouge_scorer
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_model(model, tokenizer, eval_data, device="cpu"):
    """Generate on eval_data and return BLEU + ROUGE scores."""
    preds, refs = [], []
    for ex in eval_data:
        inputs = tokenizer(ex["prompt"], return_tensors="pt").to(device)
        out    = model.generate(**inputs, max_new_tokens=128)
        text   = tokenizer.decode(out[0], skip_special_tokens=True)
        preds.append(text)
        refs.append(ex["reference"])
    bleu_res = sacrebleu.corpus_bleu(preds, [refs]).score
    scorer     = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)
    scores     = [scorer.score(r, p) for r, p in zip(refs, preds)]
    rouge1_avg = np.mean([s["rouge1"].fmeasure for s in scores])
    rougel_avg = np.mean([s["rougeL"].fmeasure for s in scores])
    return {"BLEU": bleu_res, "ROUGE-1": rouge1_avg, "ROUGE-L": rougel_avg}

if __name__ == "__main__":
    base_dir   = Path(__file__).resolve().parent.parent
    model_path = base_dir / "qlora_mistral_output" / "final"
    data_path  = base_dir / "data" / "eval_prompts.json"
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = (
        AutoModelForCausalLM
        .from_pretrained(model_path, device_map="auto", local_files_only=True)
        .to(device).eval()
    )

    with open(data_path, "r") as f:
        eval_data = json.load(f)

    metrics = evaluate_model(model, tokenizer, eval_data, device=device)
    print("\n📊 Evaluation Results:")
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")
