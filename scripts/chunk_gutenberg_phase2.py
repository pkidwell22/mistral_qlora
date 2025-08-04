# chunk_gutenberg_phase2_streamed.py

from datasets import load_from_disk, Dataset, concatenate_datasets
from pathlib import Path
import time

# ─── Config ─────────────────────────────────────────────────────────────
INTERMEDIATE_PATH = Path(r"C:/Users/pkidw/PycharmProjects/mistral_qlora/local_datasets/temp_chunked_gutenberg")
FINAL_OUTPUT_PATH = Path(r"C:/Users/pkidw/PycharmProjects/mistral_qlora/local_datasets/project_gutenberg_chunks")
BATCH_SIZE = 10_000   # Number of input_ids per saved file
MERGE_FINAL = False   # Set to True to merge parts after chunking

# ─── Setup ──────────────────────────────────────────────────────────────
start_time = time.time()
output_dir = FINAL_OUTPUT_PATH
output_dir.mkdir(parents=True, exist_ok=True)

# ─── Load Dataset ───────────────────────────────────────────────────────
print(f"📂 Loading dataset from: {INTERMEDIATE_PATH}")
dataset = load_from_disk(str(INTERMEDIATE_PATH))

print(f"📦 Flattening 'input_ids' in batches of {BATCH_SIZE}...")

# ─── Streaming Flatten + Save ───────────────────────────────────────────
flat_input_ids = []
total_written = 0
part = 0

for i, example in enumerate(dataset):
    flat_input_ids.extend(example["input_ids"])

    if len(flat_input_ids) >= BATCH_SIZE:
        part_dataset = Dataset.from_dict({"input_ids": flat_input_ids})
        part_path = output_dir / f"part_{part}"
        part_dataset.save_to_disk(str(part_path))
        print(f"💾 Saved {len(flat_input_ids):,} input_ids → {part_path.name}")
        total_written += len(flat_input_ids)
        flat_input_ids = []
        part += 1

# ─── Final Flush ────────────────────────────────────────────────────────
if flat_input_ids:
    part_dataset = Dataset.from_dict({"input_ids": flat_input_ids})
    part_path = output_dir / f"part_{part}"
    part_dataset.save_to_disk(str(part_path))
    print(f"💾 Saved final {len(flat_input_ids):,} input_ids → {part_path.name}")
    total_written += len(flat_input_ids)

print(f"✅ Finished saving {total_written:,} input_ids across {part + 1} parts in {time.time() - start_time:.2f}s.")

# ─── Optional: Merge All ────────────────────────────────────────────────
if MERGE_FINAL:
    print("🧪 Merging parts into single dataset...")
    parts = [load_from_disk(str(output_dir / f"part_{i}")) for i in range(part + 1)]
    merged = concatenate_datasets(parts)
    merged_path = output_dir / "merged"
    merged.save_to_disk(str(merged_path))
    print(f"🎉 Merged dataset saved to: {merged_path}")
