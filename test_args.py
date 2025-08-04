from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./test_output",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-4,
    evaluation_strategy="epoch"
)

print("✅ TrainingArguments loaded successfully.")
