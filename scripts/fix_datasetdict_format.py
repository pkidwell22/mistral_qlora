from datasets import load_from_disk, DatasetDict

for name in ["gutenberg_en", "openorca"]:
    print(f"🔁 Patching {name}...")
    ds_path = f"local_datasets/{name}"
    ds = load_from_disk(ds_path)

    if not isinstance(ds, DatasetDict):
        # Assume this is a plain Dataset, wrap it
        ds = DatasetDict({"train": ds})
        ds.save_to_disk(ds_path)
        print(f"✅ Re-saved {name} as DatasetDict")
    else:
        print(f"✅ {name} already ok")
