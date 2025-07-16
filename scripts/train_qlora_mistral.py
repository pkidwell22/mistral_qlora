# File: train_qlora_mistral.py

import os, torch, argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split

def format_prompt(example):
    return {
        "text": f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['act']}"
    }

def tokenize(example, tokenizer, max_length):
    return tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=max_length
    )

def main():
    parser = argparse.ArgumentParser(description="Train Mistral 7B with QLoRA")
    parser.add_argument("--data_path", default="prompts.csv", help="CSV with prompt/act columns")
    parser.add_argument("--output_dir", default="qlora_mistral_output", help="Where to save adapters")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=2)
    args = parser.parse_args()

    dataset = load_dataset("csv", data_files=args.data_path)["train"]
    dataset = dataset.map(format_prompt)

    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_data = train_data.map(lambda e: tokenize(e, tokenizer, args.max_length), batched=True)
    val_data = val_data.map(lambda e: tokenize(e, tokenizer, args.max_length), batched=True)

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
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    import transformers
    print("TrainingArguments source:", transformers.TrainingArguments.__module__)

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
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
