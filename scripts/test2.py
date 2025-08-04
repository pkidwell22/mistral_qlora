# scripts/test_dataset_load.py
from datasets import load_from_disk

for name in ["hle", "dolly", "gutenberg_en", "openorca"]:
    try:
        ds = load_from_disk(f"local_datasets/{name}" if name != "hle" else f"data/{name}")
        print(f"✅ Loaded: {name} – splits: {list(ds.keys())}")
    except Exception as e:
        print(f"❌ Failed: {name} – {e}")
