#!/usr/bin/env python3
"""
Aggregate high-quality Python/LLM datasets from HuggingFace Hub
Creates a curated dataset for fine-tuning an LLM coding assistant
"""

from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import random
from typing import List, Dict
import re


class HFDatasetAggregator:
    """Aggregate and process datasets from HuggingFace"""

    def __init__(self):
        self.datasets = []
        self.processed_examples = []

    def load_code_alpaca(self, max_samples: int = 2000):
        """Load Code Alpaca dataset - Python instruction-following"""
        print("Loading Code Alpaca dataset...")
        try:
            dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

            # Filter for Python examples
            python_examples = []
            for example in dataset:
                if "python" in example.get("instruction", "").lower() or \
                        "def " in example.get("output", "") or \
                        "import " in example.get("output", ""):
                    python_examples.append({
                        "instruction": example["instruction"],
                        "input": example.get("input", ""),
                        "output": example["output"]
                    })

                if len(python_examples) >= max_samples:
                    break

            print(f"  Loaded {len(python_examples)} Python examples from Code Alpaca")
            self.datasets.extend(python_examples)

        except Exception as e:
            print(f"  Error loading Code Alpaca: {e}")

    def load_evol_instruct_code(self, max_samples: int = 2000):
        """Load Evol-Instruct-Code dataset"""
        print("Loading Evol-Instruct-Code dataset...")
        try:
            dataset = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")

            examples = []
            for idx, example in enumerate(dataset):
                if idx >= max_samples:
                    break

                examples.append({
                    "instruction": example["instruction"],
                    "input": "",
                    "output": example["output"]
                })

            print(f"  Loaded {len(examples)} examples from Evol-Instruct-Code")
            self.datasets.extend(examples)

        except Exception as e:
            print(f"  Error loading Evol-Instruct-Code: {e}")

    def load_python_code_instructions(self, max_samples: int = 1500):
        """Load Python code instruction dataset"""
        print("Loading Python code instructions...")
        try:
            dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

            examples = []
            for idx, example in enumerate(dataset):
                if idx >= max_samples:
                    break

                # Clean up the output
                output = example.get("output", "")
                if output and not output.startswith("```"):
                    # Add code blocks if missing
                    output = f"```python\n{output}\n```"

                examples.append({
                    "instruction": example["instruction"],
                    "input": example.get("input", ""),
                    "output": output
                })

            print(f"  Loaded {len(examples)} examples from Python code instructions")
            self.datasets.extend(examples)

        except Exception as e:
            print(f"  Error loading Python code instructions: {e}")

    def load_self_instruct(self, max_samples: int = 1000):
        """Load self-instruct dataset filtering for programming"""
        print("Loading Self-Instruct dataset...")
        try:
            dataset = load_dataset("yizhongw/self_instruct", "self_instruct", split="train")

            programming_keywords = [
                "python", "code", "function", "program", "script", "algorithm",
                "class", "method", "debug", "error", "import", "library"
            ]

            examples = []
            for example in dataset:
                # Check if it's programming related
                text = (example.get("instruction", "") + " " + example.get("output", "")).lower()
                if any(keyword in text for keyword in programming_keywords):
                    examples.append({
                        "instruction": example["instruction"],
                        "input": example.get("instances", [{}])[0].get("input", "") if example.get("instances") else "",
                        "output": example.get("instances", [{}])[0].get("output", "") if example.get(
                            "instances") else ""
                    })

                if len(examples) >= max_samples:
                    break

            print(f"  Loaded {len(examples)} programming examples from Self-Instruct")
            self.datasets.extend(examples)

        except Exception as e:
            print(f"  Error loading Self-Instruct: {e}")

    def load_transformers_tasks(self, max_samples: int = 1000):
        """Load examples specific to transformers/PEFT/training"""
        print("Creating Transformers/PEFT specific examples...")

        # These could come from a specialized dataset, but let's create some
        transformers_examples = [
            {
                "instruction": "How do I load a model with 4-bit quantization using transformers?",
                "input": "",
                "output": """To load a model with 4-bit quantization using transformers:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    quantization_config=bnb_config,
    device_map="auto"
)
```"""
            },
            {
                "instruction": "Write a function to prepare a dataset for causal language modeling",
                "input": "",
                "output": """Here's a function to prepare a dataset for causal language modeling:

```python
def prepare_dataset_for_clm(dataset, tokenizer, max_length=512):
    def tokenize_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

        # For CLM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset
```"""
            },
            {
                "instruction": "How do I implement gradient accumulation in a training loop?",
                "input": "",
                "output": """To implement gradient accumulation in a training loop:

```python
from transformers import Trainer, TrainingArguments

# Using TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
    # This gives an effective batch size of 4 * 8 = 32
)

# Manual implementation
accumulation_steps = 8
optimizer.zero_grad()

for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps  # Scale loss
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```"""
            }
        ]

        # Add more examples as needed...
        self.datasets.extend(transformers_examples[:max_samples])
        print(f"  Added {len(transformers_examples[:max_samples])} Transformers/PEFT examples")

    def load_stack_exchange(self, max_samples: int = 1000):
        """Load Python questions from Stack Exchange"""
        print("Loading Stack Exchange Python dataset...")
        try:
            # This dataset contains Python Q&A from Stack Overflow
            dataset = load_dataset("kensho-technologies/pythonqa", split="train")

            examples = []
            for idx, example in enumerate(dataset):
                if idx >= max_samples:
                    break

                # Format as instruction-following
                examples.append({
                    "instruction": example.get("question", ""),
                    "input": example.get("context", ""),
                    "output": example.get("answer", "")
                })

            print(f"  Loaded {len(examples)} examples from Stack Exchange")
            self.datasets.extend(examples)

        except Exception as e:
            print(f"  Error loading Stack Exchange: {e}")
            # Try alternative
            self._load_alternative_qa_dataset(max_samples)

    def _load_alternative_qa_dataset(self, max_samples: int):
        """Load alternative Q&A dataset"""
        try:
            dataset = load_dataset("neulab/conala", "curated", split="train")

            examples = []
            for idx, example in enumerate(dataset):
                if idx >= max_samples // 2:  # Use fewer since it's smaller
                    break

                examples.append({
                    "instruction": example["intent"],
                    "input": "",
                    "output": f"```python\n{example['snippet']}\n```"
                })

            print(f"  Loaded {len(examples)} examples from CoNaLa")
            self.datasets.extend(examples)

        except Exception as e:
            print(f"  Error loading alternative dataset: {e}")

    def process_and_clean(self):
        """Process and clean all loaded examples"""
        print("\nProcessing and cleaning examples...")

        cleaned_examples = []

        for example in self.datasets:
            # Skip if missing required fields
            if not example.get("instruction") or not example.get("output"):
                continue

            # Clean instruction
            instruction = example["instruction"].strip()

            # Clean output
            output = example["output"].strip()

            # Ensure code blocks are properly formatted
            if "```" not in output and any(keyword in instruction.lower()
                                           for keyword in ["code", "function", "write", "implement"]):
                # Try to detect Python code and wrap it
                lines = output.split('\n')
                code_started = False
                new_lines = []

                for line in lines:
                    if (line.strip().startswith(('def ', 'class ', 'import ', 'from ')) or
                            (code_started and line.startswith((' ', '\t')))):
                        if not code_started:
                            new_lines.append("```python")
                            code_started = True
                        new_lines.append(line)
                    else:
                        if code_started and line.strip() and not line.startswith((' ', '\t')):
                            new_lines.append("```")
                            code_started = False
                        new_lines.append(line)

                if code_started:
                    new_lines.append("```")

                output = '\n'.join(new_lines)

            # Format as single text for training
            if example.get("input"):
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{example['input']}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

            cleaned_examples.append({
                "instruction": instruction,
                "input": example.get("input", ""),
                "output": output,
                "text": text
            })

        self.processed_examples = cleaned_examples
        print(f"Cleaned {len(self.processed_examples)} examples")

    def create_dataset(self, train_ratio: float = 0.95):
        """Create final dataset with train/validation split"""
        if not self.processed_examples:
            self.process_and_clean()

        # Shuffle
        random.shuffle(self.processed_examples)

        # Split
        split_idx = int(len(self.processed_examples) * train_ratio)
        train_examples = self.processed_examples[:split_idx]
        val_examples = self.processed_examples[split_idx:]

        # Create datasets
        train_dataset = Dataset.from_list(train_examples)
        val_dataset = Dataset.from_list(val_examples)

        # Create DatasetDict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })

        return dataset_dict

    def save_dataset(self, output_path: str = "./llm_python_combined_dataset"):
        """Save the combined dataset"""
        dataset = self.create_dataset()

        # Save in HF format
        dataset.save_to_disk(output_path)

        # Save as JSONL
        dataset["train"].to_json(f"{output_path}/train.jsonl")
        dataset["validation"].to_json(f"{output_path}/validation.jsonl")

        print(f"\nDataset saved to {output_path}")
        print(f"Train examples: {len(dataset['train'])}")
        print(f"Validation examples: {len(dataset['validation'])}")

        return dataset


def main():
    """Create combined dataset from HuggingFace sources"""

    aggregator = HFDatasetAggregator()

    # Load various datasets (adjust max_samples as needed)
    aggregator.load_code_alpaca(max_samples=3000)
    aggregator.load_evol_instruct_code(max_samples=2000)
    aggregator.load_python_code_instructions(max_samples=2000)
    aggregator.load_self_instruct(max_samples=1000)
    aggregator.load_transformers_tasks(max_samples=500)
    aggregator.load_stack_exchange(max_samples=1500)

    # Create and save dataset
    dataset = aggregator.save_dataset()

    # Show sample
    print("\nSample from training set:")
    print(dataset["train"][0]["text"][:500] + "...")

    # Show statistics
    print("\nDataset statistics:")
    print(f"Total unique instructions: {len(set(ex['instruction'] for ex in dataset['train']))}")

    # Quick quality check
    code_examples = sum(1 for ex in dataset['train'] if '```' in ex['output'])
    print(f"Examples with code blocks: {code_examples} ({code_examples / len(dataset['train']) * 100:.1f}%)")


if __name__ == "__main__":
    main()