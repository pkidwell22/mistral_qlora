import os
import math
import argparse
import wandb
from pathlib import Path

from peft import LoraConfig, get_peft_model
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from datasets import load_from_disk

# ─── Main ───
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="qlora_outputs/base_gutenberg_2ep")
    parser.add_argument("--model_name", default="hf_models/mistral-7b")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    run_name = os.path.basename(args.output_dir.rstrip("/"))
    wandb.init(project="mistral-qlora", name=run_name, config=vars(args))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )

    model_path = Path(args.model_name)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model_path = model_path.resolve()

    if not model_path.exists():
        raise ValueError(f"❌ Model path does not exist: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path.as_posix(), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path.as_posix(),
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        local_files_only=True,
    )

    if not args.mixed:
        raise ValueError("Non-mixed mode not implemented. Use --mixed")

    print("📚 Loading Gutenberg dataset only...")

    dataset_path = Path("local_datasets/gutenberg_base_1B").resolve()
    dataset = load_from_disk(dataset_path.as_posix())

    # ─── Tokenization ───
    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }

    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_data = split["train"].map(tokenize, batched=True, remove_columns=split["train"].column_names)
    val_data = split["test"].map(tokenize, batched=True, remove_columns=split["test"].column_names)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        logging_steps=10,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=["wandb"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    if args.eval_only:
        metrics = trainer.evaluate()
    else:
        trainer.train()
        metrics = trainer.evaluate()

    loss = metrics.get("eval_loss")
    perplexity = math.exp(loss) if loss else float("inf")
    print(f"📉 Eval loss: {loss:.4f}  |  Perplexity: {perplexity:.2f}")
    wandb.log({"eval_loss": loss, "perplexity": perplexity})

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Saved adapter + tokenizer to: {args.output_dir}")
