# scripts/scan_dataset_tokens_and_size.py

import os
import json
from pathlib import Path
from transformers import AutoTokenizer

def stream_texts(file_path):
    try:
        if file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    yield line.strip()

        elif file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield str(item.get("text", item))
                elif isinstance(data, dict):
                    yield str(data.get("text", data))

        elif file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        yield obj.get("text", line.strip())
                    except:
                        continue
    except Exception as e:
        print(f"❌ Couldn’t read {file_path.name}: {e}")

def scan_dataset(folder, tokenizer):
    total_tokens = 0
    total_bytes = 0
    total_files = 0

    for path in Path(folder).rglob("*"):
        if path.is_file() and path.suffix in [".txt", ".json", ".jsonl"]:
            total_files += 1
            try:
                total_bytes += os.path.getsize(path)
            except:
                continue

            for text in stream_texts(path):
                tokens = tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)

    return total_tokens, total_bytes, total_files

if __name__ == "__main__":
    dataset_dir = "local_datasets"
    tokenizer_path = "./tokenizer_mistral"  # Point to your copied folder

    print(f"\n🔍 Scanning: {dataset_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        local_files_only=True
    )

    tokens, bytes_, files = scan_dataset(dataset_dir, tokenizer)

    mb = bytes_ / 1024 / 1024
    gb = mb / 1024

    print(f"\n📁 Total files: {files}")
    print(f"💾 Total size: {mb:,.2f} MB ({gb:,.2f} GB)")
    print(f"🔢 Total tokens: {tokens:,}\n")
