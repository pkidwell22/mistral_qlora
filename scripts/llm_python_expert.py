#!/usr/bin/env python3
"""
Create a high-quality HuggingFace dataset for training a Python/LLM development expert adapter
This creates a dataset that can be directly loaded with datasets.load_dataset()
"""

from datasets import Dataset, DatasetDict, Features, Value
import json
from pathlib import Path
from typing import List, Dict
import random


class LLMPythonHFDataset:
    """Create HuggingFace dataset for LLM Python development"""

    def __init__(self):
        self.examples = []

    def add_example(self, instruction: str, input: str = "", output: str = ""):
        """Add a training example"""
        self.examples.append({
            "instruction": instruction,
            "input": input,
            "output": output,
            # Add a formatted version for easier training
            "text": self._format_example(instruction, input, output)
        })

    def _format_example(self, instruction: str, input: str, output: str) -> str:
        """Format example as a single text string for training"""
        if input:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    def create_dataset(self):
        """Create comprehensive dataset for LLM Python development"""

        # 1. Basic model loading
        self.add_example(
            "Write code to load a pretrained language model with transformers",
            "",
            """```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Model loaded: {model.config._name_or_path}")
print(f"Model size: {model.num_parameters():,} parameters")
```"""
        )

        # 2. QLoRA configuration
        self.add_example(
            "Set up QLoRA fine-tuning with 4-bit quantization",
            "",
            """```python
from transformers import BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```"""
        )

        # 3. Dataset preparation with HF datasets
        self.add_example(
            "Load and prepare a dataset using Hugging Face datasets",
            "Dataset in JSONL format with instruction/response pairs",
            """```python
from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling

def prepare_dataset(file_path: str, tokenizer, max_length: int = 512):
    \"\"\"Load and prepare dataset for training\"\"\"

    # Load dataset
    dataset = load_dataset('json', data_files=file_path, split='train')

    def formatting_func(examples):
        \"\"\"Format examples into training texts\"\"\"
        texts = []
        for instruction, input_text, output in zip(
            examples["instruction"], 
            examples.get("input", [""] * len(examples["instruction"])), 
            examples["output"]
        ):
            if input_text:
                text = f"### Instruction:\\n{instruction}\\n\\n### Input:\\n{input_text}\\n\\n### Response:\\n{output}"
            else:
                text = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{output}"
            texts.append(text)
        return {"text": texts}

    # Format the dataset
    dataset = dataset.map(
        formatting_func,
        batched=True,
        remove_columns=dataset.column_names
    )

    def tokenize_function(examples):
        \"\"\"Tokenize the texts\"\"\"
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    return tokenized_dataset

# Usage
train_dataset = prepare_dataset("train.jsonl", tokenizer)
print(f"Dataset size: {len(train_dataset)}")
print(f"First example keys: {train_dataset[0].keys()}")
```"""
        )

        # 4. Efficient data loading
        self.add_example(
            "Create an efficient data loader with streaming for large datasets",
            "",
            """```python
from datasets import load_dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

def create_streaming_dataset(file_path: str, tokenizer, max_length: int = 512):
    \"\"\"Create a streaming dataset for very large files\"\"\"

    # Load as iterable dataset
    dataset = load_dataset(
        'json', 
        data_files=file_path, 
        split='train',
        streaming=True  # Enable streaming
    )

    def preprocess_function(examples):
        \"\"\"Preprocess and tokenize examples\"\"\"
        # Format text
        if examples.get("input"):
            text = f"### Instruction:\\n{examples['instruction']}\\n\\n### Input:\\n{examples['input']}\\n\\n### Response:\\n{examples['output']}"
        else:
            text = f"### Instruction:\\n{examples['instruction']}\\n\\n### Response:\\n{examples['output']}"

        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        # Set labels
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Apply preprocessing
    dataset = dataset.map(preprocess_function)

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=data_collator,
        shuffle=False  # Streaming datasets can't be shuffled globally
    )

    return dataloader

# Usage
train_dataloader = create_streaming_dataset("large_dataset.jsonl", tokenizer)
```"""
        )

        # 5. Training with Trainer
        self.add_example(
            "Set up training using the Hugging Face Trainer",
            "",
            """```python
from transformers import Trainer, TrainingArguments
from peft import TaskType

def setup_training(model, tokenizer, train_dataset, eval_dataset=None):
    \"\"\"Set up training with HF Trainer\"\"\"

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        learning_rate=2e-4,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        push_to_hub=False,
        report_to=["tensorboard"],
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        ),
    )

    # Disable cache for gradient checkpointing
    model.config.use_cache = False

    return trainer

# Usage
trainer = setup_training(model, tokenizer, train_dataset, eval_dataset)
trainer.train()
```"""
        )

        # 6. Custom dataset class
        self.add_example(
            "Create a custom dataset class for specialized data handling",
            "",
            """```python
from torch.utils.data import Dataset
import torch
from datasets import Dataset as HFDataset

class CodeInstructionDataset(Dataset):
    \"\"\"Custom dataset for code instruction pairs\"\"\"

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format the prompt
        if item.get('input'):
            prompt = f"### Instruction:\\n{item['instruction']}\\n\\n### Input:\\n{item['input']}\\n\\n### Response:\\n{item['output']}"
        else:
            prompt = f"### Instruction:\\n{item['instruction']}\\n\\n### Response:\\n{item['output']}"

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Prepare labels
        labels = encoding["input_ids"].clone()

        # Find response start position to mask instruction from loss
        response_start_token = "### Response:"
        response_tokens = self.tokenizer.encode(response_start_token, add_special_tokens=False)

        # Find where response starts
        input_ids_list = encoding["input_ids"][0].tolist()
        response_pos = None
        for i in range(len(input_ids_list) - len(response_tokens)):
            if input_ids_list[i:i+len(response_tokens)] == response_tokens:
                response_pos = i + len(response_tokens)
                break

        # Mask everything before response
        if response_pos:
            labels[0, :response_pos] = -100

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": labels[0]
        }

# Convert to HF Dataset for compatibility
def custom_to_hf_dataset(custom_dataset):
    \"\"\"Convert custom dataset to HF dataset\"\"\"
    data_dict = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for i in range(len(custom_dataset)):
        item = custom_dataset[i]
        for key in data_dict:
            data_dict[key].append(item[key].numpy())

    return HFDataset.from_dict(data_dict)

# Usage
custom_dataset = CodeInstructionDataset("train.jsonl", tokenizer)
hf_dataset = custom_to_hf_dataset(custom_dataset)
```"""
        )

        # 7. Multi-dataset training
        self.add_example(
            "Combine multiple datasets for training",
            "",
            """```python
from datasets import concatenate_datasets, interleave_datasets

def combine_datasets_for_training(dataset_paths: Dict[str, float], tokenizer):
    \"\"\"Combine multiple datasets with optional weighting\"\"\"

    datasets = []
    weights = []

    for path, weight in dataset_paths.items():
        # Load dataset
        ds = load_dataset('json', data_files=path, split='train')

        # Preprocess
        def preprocess(examples):
            texts = []
            for i in range(len(examples['instruction'])):
                inst = examples['instruction'][i]
                inp = examples.get('input', [''] * len(examples['instruction']))[i]
                out = examples['output'][i]

                if inp:
                    text = f"### Instruction:\\n{inst}\\n\\n### Input:\\n{inp}\\n\\n### Response:\\n{out}"
                else:
                    text = f"### Instruction:\\n{inst}\\n\\n### Response:\\n{out}"
                texts.append(text)

            # Tokenize
            model_inputs = tokenizer(texts, max_length=512, truncation=True, padding="max_length")
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs

        ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
        datasets.append(ds)
        weights.append(weight)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w/total_weight for w in weights]

    # Interleave datasets with weights
    combined_dataset = interleave_datasets(
        datasets,
        probabilities=weights,
        seed=42
    )

    return combined_dataset

# Usage
dataset_paths = {
    "python_code.jsonl": 0.5,      # 50% Python code
    "llm_instructions.jsonl": 0.3,  # 30% LLM instructions  
    "general_qa.jsonl": 0.2         # 20% general Q&A
}

combined_dataset = combine_datasets_for_training(dataset_paths, tokenizer)
print(f"Combined dataset size: {len(combined_dataset)}")
```"""
        )

        # 8. Evaluation during training
        self.add_example(
            "Implement custom evaluation metrics during training",
            "",
            """```python
from transformers import TrainerCallback
import evaluate
import numpy as np

class CustomEvalCallback(TrainerCallback):
    \"\"\"Custom evaluation callback for monitoring specific metrics\"\"\"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        \"\"\"Called after evaluation phase\"\"\"
        print(f"\\nEvaluation at step {state.global_step}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

def compute_metrics(eval_preds, tokenizer):
    \"\"\"Compute custom metrics for evaluation\"\"\"
    predictions, labels = eval_preds

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean up
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute ROUGE scores
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # Extract scores
    results = {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
    }

    return results

# Custom Trainer with evaluation
class EvaluatingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        \"\"\"Custom loss computation with additional logging\"\"\"
        outputs = model(**inputs)
        loss = outputs.loss

        # Log additional metrics
        if self.state.global_step % self.args.logging_steps == 0:
            with torch.no_grad():
                perplexity = torch.exp(loss)
                self.log({"perplexity": perplexity.item()})

        return (loss, outputs) if return_outputs else loss

# Usage
trainer = EvaluatingTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=lambda p: compute_metrics(p, tokenizer),
    callbacks=[CustomEvalCallback(tokenizer)]
)
```"""
        )

        # Add more examples...
        self._add_more_examples()

    def _add_more_examples(self):
        """Add additional specialized examples"""

        # Add inference optimization
        self.add_example(
            "Optimize model inference for production deployment",
            "",
            """```python
import torch
from transformers import TextStreamer

class OptimizedInference:
    \"\"\"Optimized inference pipeline for production\"\"\"

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # Enable eval mode
        self.model.eval()

        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            self.model = torch.compile(model)

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 256, stream: bool = False):
        \"\"\"Generate text with optimizations\"\"\"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Streaming
        if stream:
            streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                streamer=streamer,
            )
        else:
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )

        # Generate
        outputs = self.model.generate(**generation_kwargs)

        if not stream:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt from response
            response = response[len(prompt):].strip()
            return response

    def batch_generate(self, prompts: List[str], max_new_tokens: int = 256):
        \"\"\"Batch generation for efficiency\"\"\"

        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.cuda.amp.autocast():  # Mixed precision
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Remove prompts
        cleaned_responses = []
        for prompt, response in zip(prompts, responses):
            cleaned = response[len(prompt):].strip()
            cleaned_responses.append(cleaned)

        return cleaned_responses

# Usage
inference = OptimizedInference(model, tokenizer)
response = inference.generate("Write a Python function to sort a list")
```"""
        )

        # Dataset validation
        self.add_example(
            "Validate and clean a training dataset",
            "",
            """```python
from datasets import load_dataset
import re
from typing import Dict, List

class DatasetValidator:
    \"\"\"Validate and clean training datasets\"\"\"

    def __init__(self, min_length: int = 10, max_length: int = 2048):
        self.min_length = min_length
        self.max_length = max_length
        self.stats = {
            "total": 0,
            "filtered": 0,
            "reasons": {}
        }

    def validate_example(self, example: Dict) -> bool:
        \"\"\"Validate a single example\"\"\"

        # Check required fields
        if not all(key in example for key in ["instruction", "output"]):
            self._add_filter_reason("missing_fields")
            return False

        # Check lengths
        inst_len = len(example["instruction"].split())
        out_len = len(example["output"].split())

        if inst_len < 3:
            self._add_filter_reason("instruction_too_short")
            return False

        if out_len < self.min_length:
            self._add_filter_reason("output_too_short")
            return False

        if out_len > self.max_length:
            self._add_filter_reason("output_too_long")
            return False

        # Check for code blocks if instruction mentions code
        if any(word in example["instruction"].lower() for word in ["code", "function", "script", "program"]):
            if "```" not in example["output"]:
                self._add_filter_reason("missing_code_blocks")
                return False

        # Check for repetitive content
        words = example["output"].split()
        if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
            self._add_filter_reason("too_repetitive")
            return False

        return True

    def _add_filter_reason(self, reason: str):
        \"\"\"Track filtering reasons\"\"\"
        self.stats["filtered"] += 1
        self.stats["reasons"][reason] = self.stats["reasons"].get(reason, 0) + 1

    def clean_dataset(self, dataset_path: str) -> Dataset:
        \"\"\"Clean and validate dataset\"\"\"

        # Load dataset
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        self.stats["total"] = len(dataset)

        # Filter
        cleaned_dataset = dataset.filter(self.validate_example)

        # Print statistics
        print(f"Dataset Validation Statistics:")
        print(f"  Total examples: {self.stats['total']}")
        print(f"  Filtered: {self.stats['filtered']}")
        print(f"  Kept: {len(cleaned_dataset)}")
        print(f"\\nFilter reasons:")
        for reason, count in self.stats["reasons"].items():
            print(f"  {reason}: {count}")

        return cleaned_dataset

# Usage
validator = DatasetValidator(min_length=20, max_length=1024)
clean_dataset = validator.clean_dataset("raw_dataset.jsonl")
clean_dataset.save_to_disk("cleaned_dataset")
```"""
        )

    def to_huggingface_dataset(self) -> DatasetDict:
        """Convert to HuggingFace DatasetDict"""
        # Shuffle examples
        random.shuffle(self.examples)

        # Split into train/validation
        val_size = int(len(self.examples) * 0.1)
        train_examples = self.examples[val_size:]
        val_examples = self.examples[:val_size]

        # Create datasets
        train_dataset = Dataset.from_list(train_examples)
        val_dataset = Dataset.from_list(val_examples)

        # Create DatasetDict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })

        return dataset_dict

    def save_to_hub(self, repo_name: str, private: bool = True):
        """Save dataset to Hugging Face Hub"""
        dataset_dict = self.to_huggingface_dataset()
        dataset_dict.push_to_hub(repo_name, private=private)
        print(f"Dataset pushed to hub: {repo_name}")

    def save_locally(self, path: str):
        """Save dataset locally in HF format"""
        dataset_dict = self.to_huggingface_dataset()
        dataset_dict.save_to_disk(path)
        print(f"Dataset saved to: {path}")

        # Also save as JSONL for compatibility
        for split in ["train", "validation"]:
            jsonl_path = Path(path) / f"{split}.jsonl"
            dataset_dict[split].to_json(jsonl_path)
            print(f"  {split} saved as JSONL: {jsonl_path}")


def main():
    """Create the dataset"""
    creator = LLMPythonHFDataset()
    creator.create_dataset()

    # Save locally
    creator.save_locally("./llm_python_expert_dataset")

    # Print info
    dataset_dict = creator.to_huggingface_dataset()
    print("\nDataset created successfully!")
    print(f"Train examples: {len(dataset_dict['train'])}")
    print(f"Validation examples: {len(dataset_dict['validation'])}")

    # Show how to load
    print("\nTo load this dataset:")
    print("from datasets import load_from_disk")
    print("dataset = load_from_disk('./llm_python_expert_dataset')")
    print("\nOr load individual splits:")
    print("from datasets import load_dataset")
    print("dataset = load_dataset('json', data_files={'train': 'train.jsonl', 'validation': 'validation.jsonl'})")


if __name__ == "__main__":
    main()