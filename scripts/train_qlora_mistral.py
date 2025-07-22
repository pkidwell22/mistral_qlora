import os
import math
import argparse
import wandb

from peft import LoraConfig, get_peft_model
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from datasets import load_dataset, load_from_disk, concatenate_datasets

# ─── Tokenization & Formatting ───
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

def format_pair(example, prompt_key, response_key):
    return {
        "text": f"### Instruction:\n{example.get(prompt_key, '')}\n\n### Response:\n{example.get(response_key, '')}"
    }

def load_and_format(path, formatter):
    raw = load_from_disk(path)
    if isinstance(raw, dict):
        raw = raw.get("train", next(iter(raw.values())))
    formatted = raw.map(lambda ex: formatter(ex), keep_in_memory=True)
    if "text" in formatted.column_names:
        formatted = formatted.remove_columns([col for col in formatted.column_names if col != "text"])
    return formatted.filter(lambda x: x.get("text", "").strip() != "")

def load_and_format_hermes(path):
    raw = load_from_disk(path)
    if isinstance(raw, dict):
        raw = raw.get("train", next(iter(raw.values())))
    def format_chat(example):
        try:
            return format_pair(example["conversations"][0] | example["conversations"][1], "value", "value")
        except Exception:
            return {"text": example.get("text", "")}
    return raw.map(format_chat).filter(lambda x: x.get("text", "").strip() != "")

def load_and_format_orca(path):
    raw = load_from_disk(path)
    if isinstance(raw, dict):
        raw = raw.get("train", next(iter(raw.values())))
    return raw.map(lambda ex: format_pair(ex, "question", "response"))\
              .filter(lambda x: x.get("text", "").strip() != "")

# ─── Main ───
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="qlora_mistral_output")
    parser.add_argument("--model_name", default="/mnt/c/Users/pkidw/hf_models/mistral-7b")
    parser.add_argument("--epochs", type=int, default=1)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )

    if not args.mixed:
        raise ValueError("Non-mixed mode not implemented. Use --mixed")

    print("🔀 Loading and mixing datasets: ALL available")

    datasets = []

    def safe_add(name, fn, pct):
        try:
            ds = fn()
            if ds and len(ds) > 0:
                print(f"✅ Loaded {name}: {len(ds)} rows")
                datasets.append(ds.shuffle(seed=42).select(range(int(len(ds) * pct))))
            else:
                print(f"⚠️ {name} is empty or invalid")
        except Exception as e:
            print(f"⚠️ Failed to load {name}: {e}")

    safe_add("openhermes", lambda: load_and_format_hermes("local_datasets/openhermes_2_5"), 0.20)
    safe_add("openorca", lambda: load_and_format_orca("local_datasets/openorca"), 0.20)
    safe_add("dolly", lambda: load_and_format("local_datasets/dolly", lambda ex: format_pair(ex, "instruction", "response")), 0.10)
    safe_add("hle", lambda: load_and_format("local_datasets/hle", lambda ex: format_pair(ex, "question", "answer")), 0.15)
    safe_add("moss", lambda: load_and_format("local_datasets/moss", lambda ex: format_pair(ex, "question", "mc1_targets")), 0.05)
    safe_add("gutenberg", lambda: load_from_disk("local_datasets/gutenberg_en").map(lambda ex: {"text": ex["text"]}).filter(lambda x: x["text"].strip() != ""), 0.07)
    safe_add("codealpaca", lambda: load_and_format("local_datasets/codealpaca", lambda ex: format_pair(ex, "instruction", "output")), 0.05)
    safe_add("truthfulqa", lambda: load_and_format("local_datasets/truthfulqa", lambda ex: format_pair(ex, "instruction", "output")), 0.07)
    safe_add("truthfulqa_mc", lambda: load_and_format("local_datasets/truthfulqa_mc", lambda ex: format_pair(ex, "question", "mc1_targets")), 0.07)

    if not datasets:
        raise RuntimeError("❌ No datasets loaded. Cannot proceed with training.")

    merged = concatenate_datasets(datasets).shuffle(seed=123)
    split = merged.train_test_split(test_size=0.05, seed=42)
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
