from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer
import time

# ─── Config ─────────────────────────────────────────────────────────────
DATASET_ROOT = Path(r"C:/Users/pkidw/PycharmProjects/mistral_qlora/local_datasets")
TOKENIZER_PATH = Path(r"C:/Users/pkidw/PycharmProjects/hf_tokenizer_mistral")
SKIP_DATASETS = {"project_gutenberg_chunks"}
LOG_INTERVAL = 10_000

def count_tokens(dataset, tokenizer):
    total_tokens = 0
    total_examples = 0

    for i, example in enumerate(dataset):
        if i % LOG_INTERVAL == 0 and i > 0:
            print(f"   ⏳ {i:,} examples processed...")
        total_examples += 1

        if "input_ids" in example and isinstance(example["input_ids"], list):
            total_tokens += len(example["input_ids"])
        else:
            text = (
                example.get("text")
                or next((v for v in example.values() if isinstance(v, str)), None)
            )
            if text:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)

    return total_examples, total_tokens

def count_tokens_in_dataset(dataset_path, tokenizer):
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
    print(f"\n🔍 Scanning datasets in: {DATASET_ROOT}\n")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    grand_total_examples = 0
    grand_total_tokens = 0

    for folder in sorted(DATASET_ROOT.glob("*")):
        if not folder.is_dir():
            continue
        if folder.name in SKIP_DATASETS:
            print(f"⏭️ Skipping {folder.name}\n")
            continue

        print(f"📂 {folder.name}")
        ex, tok = count_tokens_in_dataset(folder, tokenizer)
        grand_total_examples += ex
        grand_total_tokens += tok
        print(f"   ➤ {ex:,} examples → {tok:,} tokens\n")

    print("📊 Final Totals (excluding Gutenberg):")
    print(f"   ✅ Total examples: {grand_total_examples:,}")
    print(f"   🔢 Total tokens: {grand_total_tokens:,}\n")
