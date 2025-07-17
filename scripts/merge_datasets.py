# File: scripts/merge_datasets.py

import pandas as pd
import os

# List your input CSV files here
csv_files = [
    "data/prompts.csv",
    "data/alpaca_train.csv",  # rename if needed
    "data/other_dataset.csv",  # add/remove as needed
]

# Load and concatenate
all_dfs = []
for path in csv_files:
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "prompt" in df.columns and "act" in df.columns:
            all_dfs.append(df[["prompt", "act"]])
        else:
            print(f"⚠️ Skipping {path}: missing required columns.")
    else:
        print(f"⚠️ File not found: {path}")

# Combine & save
if all_dfs:
    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_csv("data/merged_prompts.csv", index=False)
    print(f"✅ Merged dataset saved to: data/merged_prompts.csv")
else:
    print("❌ No valid datasets were merged.")
