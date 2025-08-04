# chunk_gutenberg_phase1.py

import os
from datasets import load_from_disk
from transformers import AutoTokenizer

# ─── Config ─────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\pkidw\PycharmProjects\mistral_qlora\local_datasets\project_gutenberg"
INTERMEDIATE_PATH = r"C:\Users\pkidw\PycharmProjects\mistral_qlora\local_datasets\temp_chunked_gutenberg"
TOKENIZER_PATH = r"C:\Users\pkidw\PycharmProjects\hf_tokenizer_mistral"
CHUNK_SIZE = 512

# ─── Load ───────────────────────────────────────────────────────────────
print(f"📂 Loading dataset from: {DATASET_PATH}")
dataset = load_from_disk(DATASET_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

# ─── Chunk Function ─────────────────────────────────────────────────────
def chunk_example(example):
    text = example.get("text", "")
    if not isinstance(text, str):
        return {"input_ids": []}
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + CHUNK_SIZE] for i in range(0, len(tokens), CHUNK_SIZE)]
    return {"input_ids": chunks}

# ─── Apply Chunking ─────────────────────────────────────────────────────
print("🔪 Chunking text into token blocks...")
chunked = dataset.map(
    chunk_example,
    remove_columns=["text"],
    batched=False,
    desc="Chunking Gutenberg",
    num_proc=1  # safe, stable
)

# ─── Save Intermediate ──────────────────────────────────────────────────
print(f"💾 Saving intermediate chunked dataset to: {INTERMEDIATE_PATH}")
chunked.save_to_disk(INTERMEDIATE_PATH)

print("✅ Phase 1 complete.")
