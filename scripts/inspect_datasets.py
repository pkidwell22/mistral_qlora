import os
from datasets import load_from_disk

# Paths from the training script
DATASET_PATHS = {
    "openhermes": "local_datasets/openhermes_2_5",
    "openorca": "local_datasets/openorca",
    "dolly": "local_datasets/dolly",
    "hle": "local_datasets/hle",
    "moss": "local_datasets/moss",
    "gutenberg": "local_datasets/gutenberg_en",
    "codealpaca": "local_datasets/codealpaca",
    "truthfulqa": "local_datasets/truthfulqa",
    "truthfulqa_mc": "local_datasets/truthfulqa_mc",
    "awesome": "local_datasets/awesome-chatgpt-prompts",
}

def inspect_dataset(name, path):
    print(f"\n🔍 {name.upper()} — {path}")
    if not os.path.exists(path):
        print("❌ Path not found.")
        return
    try:
        ds = load_from_disk(path)
        if isinstance(ds, dict):
            for split_name, subset in ds.items():
                print(f"  📁 Split: {split_name} | Rows: {len(subset)} | Columns: {subset.column_names}")
        else:
            print(f"  📁 Single dataset | Rows: {len(ds)} | Columns: {ds.column_names}")
    except Exception as e:
        print(f"  ⚠️ Could not load dataset: {e}")
    print("  📦 Contents:", os.listdir(path))

if __name__ == "__main__":
    for name, path in DATASET_PATHS.items():
        inspect_dataset(name, path)
