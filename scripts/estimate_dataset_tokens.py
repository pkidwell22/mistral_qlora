import os
import json
import argparse
import gzip
from pathlib import Path
from transformers import AutoTokenizer

def get_file_size(path):
    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / 1024 / 1024
    return size_bytes, size_mb

def extract_texts_from_file(file_path):
    if file_path.suffix == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return [f.read()]
    elif file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [item.get("text", str(item)) for item in data]
            elif isinstance(data, dict):
                return [data.get("text", str(data))]
    elif file_path.suffix == ".jsonl":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return [json.loads(line).get("text", line) for line in f]
    return []

def scan_folder(base_dir, tokenizer):
    total_tokens = 0
    total_bytes = 0
    file_count = 0

    for path in Path(base_dir).rglob("*"):
        if path.suffix in [".txt", ".json", ".jsonl"]:
            file_count += 1
            bytes_, _ = get_file_size(path)
            total_bytes += bytes_
            texts = extract_texts_from_file(path)
            for text in texts:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)

    return total_tokens, total_bytes, file_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="local_datasets",
                        help="Path to your dataset folder (default: local_datasets)")
    parser.add_argument("--tokenizer_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                        help="Tokenizer to estimate token size")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    print(f"\n🔍 Scanning datasets in: {args.dataset_dir}")
    tokens, bytes_, files = scan_folder(args.dataset_dir, tokenizer)

    size_mb = bytes_ / 1024 / 1024
    size_gb = size_mb / 1024

    print(f"\n📁 Total files: {files}")
    print(f"💾 Total size: {size_mb:,.2f} MB ({size_gb:,.2f} GB)")
    print(f"🔢 Estimated tokens: {tokens:,}\n")

if __name__ == "__main__":
    main()
