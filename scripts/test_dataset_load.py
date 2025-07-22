from datasets import load_from_disk
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

datasets_to_test = {
    "hle": os.path.join(base_dir, "data", "hle"),
    "dolly": os.path.join(base_dir, "local_datasets", "dolly"),
    "gutenberg_en": os.path.join(base_dir, "local_datasets", "gutenberg_en"),
    "openorca": os.path.join(base_dir, "local_datasets", "openorca"),
}

for name, path in datasets_to_test.items():
    print(f"🔍 Testing: {name}")
    try:
        ds = load_from_disk(path)
        if hasattr(ds, "keys"):
            print(f"✅ Loaded '{name}' – splits: {list(ds.keys())}")
        else:
            print(f"✅ Loaded '{name}' – type: {type(ds)} (not a DatasetDict)")
    except Exception as e:
        print(f"❌ Failed to load '{name}': {e}")
