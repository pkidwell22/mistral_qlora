# scripts/scan_gutenberg_chunk_tokens.py

from pathlib import Path
from datasets import load_from_disk, Dataset
import time

# ─── Config ─────────────────────────────────────────────────────────────
GUTENBERG_DIR = Path(r"C:/Users/pkidw/PycharmProjects/mistral_qlora/local_datasets/project_gutenberg_chunks")
LOG_INTERVAL = 10_000

def count_tokens(dataset):
    total_tokens = 0
    total_examples = 0

    for i, example in enumerate(dataset):
        if i % LOG_INTERVAL == 0 and i > 0:
            print(f"   ⏳ {i:,} examples processed...")

        total_examples += 1
        if "input_ids" in example and isinstance(example["input_ids"], list):
            total_tokens += len(example["input_ids"])

    return total_examples, total_tokens

# ─── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🔍 Scanning: {GUTENBERG_DIR}\n")
    start = time.time()

    grand_total_examples = 0
    grand_total_tokens = 0

    for subfolder in sorted(GUTENBERG_DIR.glob("part_*")):
        if not subfolder.is_dir():
            continue

        print(f"📂 {subfolder.name}")
        dataset = load_from_disk(str(subfolder))
        ex, tok = count_tokens(dataset)
        grand_total_examples += ex
        grand_total_tokens += tok
        print(f"   ➤ {ex:,} examples → {tok:,} tokens\n")

    print("📊 Final Totals:")
    print(f"   ✅ Total examples: {grand_total_examples:,}")
    print(f"   🔢 Total tokens: {grand_total_tokens:,}")
    print(f"   ⏱️ Time elapsed: {time.time() - start:.2f}s\n")
