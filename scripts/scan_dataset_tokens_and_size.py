# scripts/scan_dataset_tokens_and_size.py

import os
from pathlib import Path
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import AutoTokenizer


def count_tokens(dataset, tokenizer=None, log_interval=10_000):
    total_tokens = 0
    total_examples = 0

    for i, example in enumerate(dataset):
        if i % log_interval == 0 and i > 0:
            print(f"   ⏳ Processed {i:,} examples...")

        total_examples += 1

        # Case 1: dataset already tokenized
        if "input_ids" in example and isinstance(example["input_ids"], list):
            total_tokens += len(example["input_ids"])

        # Case 2: raw text dataset
        elif tokenizer:
            text = (
                example.get("text")
                or next((v for v in example.values() if isinstance(v, str)), None)
            )
            if text:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)

    return total_examples, total_tokens


def count_tokens_in_dataset(dataset_path, tokenizer=None):
    try:
        dataset = load_from_disk(dataset_path)

        if isinstance(dataset, DatasetDict):
            total_ex, total_tok = 0, 0
            for split_name, split in dataset.items():
                print(f"   📄 Split: {split_name}")
                ex, tok = count_tokens(split, tokenizer)
                total_ex += ex
                total_tok += tok
            return total_ex, total_tok

        elif isinstance(dataset, Dataset):
            return count_tokens(dataset, tokenizer)

        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    except Exception as e:
        print(f"❌ Couldn’t read {dataset_path}: {e}")
        return 0, 0


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    dataset_root = project_root / "local_datasets"
    tokenizer_path = Path(r"C:/Users/pkidw/PycharmProjects/hf_tokenizer_mistral")

    # Datasets you want to skip temporarily (e.g., too large, will be processed later)
    SKIP_DATASETS = {"openwebtext", "OpenOrca"}

    print(f"\n🔍 Scanning all datasets in: {dataset_root}\n")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    grand_total_examples = 0
    grand_total_tokens = 0

    for folder in sorted(dataset_root.glob("*")):
        if not folder.is_dir():
            continue
        if folder.name in SKIP_DATASETS:
            print(f"⏭️ Skipping {folder.name} (in SKIP_DATASETS)\n")
            continue

        # Check for nested chunked parts like project_gutenberg_chunks/part_0/
        if any((folder / sub).is_dir() for sub in os.listdir(folder)):
            print(f"📁 {folder.name}/")
            for subfolder in sorted(folder.glob("part_*")):
                if subfolder.is_dir():
                    ex, tok = count_tokens_in_dataset(subfolder, tokenizer)
                    grand_total_examples += ex
                    grand_total_tokens += tok
                    print(f"   └── {subfolder.name}: {ex:,} examples → {tok:,} tokens")
            print()

        else:
            print(f"📂 {folder.name}")
            ex, tok = count_tokens_in_dataset(folder, tokenizer)
            grand_total_examples += ex
            grand_total_tokens += tok
            print(f"   ➤ {ex:,} examples → {tok:,} tokens\n")

    print("📊 Final Totals:")
    print(f"   ✅ Total examples: {grand_total_examples:,}")
    print(f"   🔢 Total tokens: {grand_total_tokens:,}\n")
