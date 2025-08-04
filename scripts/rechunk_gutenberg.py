import os
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm

# ─── Config ─────────────────────────────────────────────────────────────
SOURCE_PATH = "C:/Users/pkidw/PycharmProjects/mistral_qlora/local_datasets/project_gutenberg"
CHUNKED_OUTPUT_PATH = "C:/Users/pkidw/PycharmProjects/mistral_qlora/local_datasets/project_gutenberg_chunks"
TOKENIZER_PATH = "~/hf_models/mistral-7b"
CHUNK_SIZE = 512

# ─── Load ───────────────────────────────────────────────────────────────
print(f"📂 Loading dataset from: {SOURCE_PATH}")
dataset = load_from_disk(SOURCE_PATH)
tokenizer = AutoTokenizer.from_pretrained(Path(TOKENIZER_PATH).expanduser().as_posix(), local_files_only=True)

# ─── Chunk Function ──────────────────────────────────────────────────────
def chunk(example):
    tokens = tokenizer(example["text"], truncation=False)["input_ids"]
    chunks = [
        {"input_ids": tokens[i:i + CHUNK_SIZE]}
        for i in range(0, len(tokens), CHUNK_SIZE)
        if len(tokens[i:i + CHUNK_SIZE]) == CHUNK_SIZE
    ]
    return {"chunks": chunks}

# ─── Process and Save ────────────────────────────────────────────────────
os.makedirs(CHUNKED_OUTPUT_PATH, exist_ok=True)
all_chunks = []

print("🔄 Chunking dataset...")
for i, example in tqdm(enumerate(dataset), total=len(dataset)):
    chunked = chunk(example)
    for j, item in enumerate(chunked["chunks"]):
        all_chunks.append({"text": tokenizer.decode(item["input_ids"])})

# Convert to HF Dataset
chunked_dataset = Dataset.from_list(all_chunks)
print(f"📊 Saving {len(chunked_dataset)} chunks to: {CHUNKED_OUTPUT_PATH}")
chunked_dataset.save_to_disk(CHUNKED_OUTPUT_PATH)
print("✅ Done.")
