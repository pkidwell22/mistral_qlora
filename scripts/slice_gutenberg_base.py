# scripts/slice_gutenberg_base.py

from datasets import load_from_disk, concatenate_datasets
from pathlib import Path

INPUT_DIR = Path("C:/Users/pkidw/PycharmProjects/mistral_qlora/local_datasets/project_gutenberg_chunks")
OUTPUT_DIR = Path("C:/Users/pkidw/PycharmProjects/mistral_qlora/local_datasets/gutenberg_base_1B")
TARGET_EXAMPLES = 1_953_125

print(f"🔍 Reading Gutenberg parts from {INPUT_DIR}")
parts = []
total = 0

for sub in sorted(INPUT_DIR.glob("part_*")):
    ds = load_from_disk(str(sub))
    if total >= TARGET_EXAMPLES:
        break
    needed = TARGET_EXAMPLES - total
    if len(ds) > needed:
        ds = ds.select(range(needed))
    parts.append(ds)
    total += len(ds)
    print(f"   ✔ Loaded {len(ds):,} from {sub.name} (total so far: {total:,})")

final_ds = concatenate_datasets(parts)
final_ds.save_to_disk(str(OUTPUT_DIR))
print(f"\n✅ Saved {len(final_ds):,} examples to: {OUTPUT_DIR}")
