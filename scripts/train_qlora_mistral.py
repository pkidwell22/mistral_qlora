import os, torch, argparse
from datasets import load_from_disk  # ✅ switched from load_dataset
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments, Trainer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling,
)
from transformers import LlamaTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def format_prompt(example):
    return {
        "text": f"### Instruction:\n{example['question']}\n\n### Response:\n{example['answer']}\n\n### Rationale:\n{example['rationale']}"
    }

def tokenize(example, tokenizer, max_length):
    return tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=max_length
    )

def main():
    parser = argparse.ArgumentParser(description="Train Mistral 7B with QLoRA")
    parser.add_argument("--data_path", default="data/hle", help="Path to local dataset saved from HF")
    parser.add_argument("--output_dir", default="qlora_mistral_output", help="Where to save adapters")
    parser.add_argument("--model_name", default="/mnt/c/Users/pkidw/hf_models/mistral-7b", help="Path to local base model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=2)
    args = parser.parse_args()

    # ─── Load and format dataset ─────────────────────────────────────
    full_dataset = load_from_disk(args.data_path)["test"]  # ✅ Only 'test' split exists
    full_dataset = full_dataset.map(format_prompt)
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    # ─── Load tokenizer ──────────────────────────────────────────────
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name,
        local_files_only=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ─── Tokenize datasets ───────────────────────────────────────────
    train_data = train_data.map(lambda e: tokenize(e, tokenizer, args.max_length), batched=True)
    val_data = val_data.map(lambda e: tokenize(e, tokenizer, args.max_length), batched=True)

    # ─── Load model in 4-bit ─────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )

    model = prepare_model_for_kbit_training(model)

    # ─── Apply LoRA Adapters ────────────────────────────────────────
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # ─── Training configuration ─────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],  # Add wandb/tensorboard if needed
    )

    # ─── Train the model ────────────────────────────────────────────
    model.config.use_cache = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    # ─── Save adapters + tokenizer ──────────────────────────────────
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Saved LoRA adapter + tokenizer to: {args.output_dir}")

if __name__ == "__main__":
    main()
