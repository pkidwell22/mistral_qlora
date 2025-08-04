# scripts/scan_raw_dataset_size.py

import os
from pathlib import Path

def scan_folder(path):
    total_bytes = 0
    file_count = 0
    for root, _, files in os.walk(path):
        for file in files:
            full_path = os.path.join(root, file)
            try:
                total_bytes += os.path.getsize(full_path)
                file_count += 1
            except Exception as e:
                print(f"❌ Skipped {file}: {e}")
    return total_bytes, file_count

if __name__ == "__main__":
    dataset_dir = "local_datasets"
    total_bytes, file_count = scan_folder(dataset_dir)

    total_mb = total_bytes / 1024 / 1024
    total_gb = total_mb / 1024

    print(f"\n📁 Scanned folder: {dataset_dir}")
    print(f"📄 Total files: {file_count}")
    print(f"💾 Total size: {total_mb:,.2f} MB ({total_gb:,.2f} GB)\n")
