#!/usr/bin/env python3
from datasets import load_dataset

# 1) Load Alpaca-cleaned
ds = load_dataset("yahma/alpaca-cleaned")

# 2) Split off a 10% validation set
split    = ds["train"].train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
valid_ds = split["test"]

# 3) Save to JSONL (one JSON per line)
train_ds.to_json("data/alpaca_train.jsonl", orient="records", lines=True)
valid_ds.to_json("data/alpaca_valid.jsonl", orient="records", lines=True)

print(f"✅ Wrote {len(train_ds)} train + {len(valid_ds)} valid examples")
