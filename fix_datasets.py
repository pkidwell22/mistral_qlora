from datasets import Dataset, DatasetDict
import pandas as pd
import os

# Gutenberg
df_gut = pd.DataFrame([
    {"text": "Call me Ishmael."},
    {"text": "It was the best of times, it was the worst of times."}
])
ds_gut = Dataset.from_pandas(df_gut)
DatasetDict({"train": ds_gut}).save_to_disk("local_datasets/gutenberg_en")

# OpenOrca
df_orca = pd.DataFrame([{
    "system_prompt": "You are a helpful assistant.",
    "question": "What is the capital of France?",
    "response": "The capital of France is Paris."
}])
ds_orca = Dataset.from_pandas(df_orca)
DatasetDict({"train": ds_orca}).save_to_disk("local_datasets/openorca")

print("✅ Both datasets saved as DatasetDicts.")
