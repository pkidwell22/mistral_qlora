from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

# ─── Apply LoRA Adapters ───────────────────────────────────
model = get_peft_model(model, peft_config)

# ─── Patch Tokenizer Padding ───────────────────────────────
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos as pad token for causal LM

# ─── Data Collator ─────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM = no masked language modeling
)

# ─── Training Arguments ────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    logging_steps=25,
    save_steps=500,
    save_total_limit=1,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit",
    report_to="none",  # or "wandb"
    evaluation_strategy="no",
)

# ─── Start Training ────────────────────────────────────────
print("🚀 Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)
trainer.train()

# ─── Save LoRA Adapters and Tokenizer ──────────────────────
print("💾 Saving model & tokenizer...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete.")
