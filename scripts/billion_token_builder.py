#!/usr/bin/env python3
"""
1 Billion Token Dataset Builder
Combines 712M Gutenberg + 288M Conversational = 1B token hybrid base
"""

import os
import gc
import time
from pathlib import Path
from typing import List
from tqdm import tqdm
import json
import random

from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer


class BillionTokenBuilder:
    def __init__(self, output_dir="./gutenberg_1b_hybrid"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Token targets
        self.gutenberg_target = 712_000_000  # We have more than 600M!
        self.conversational_target = 288_000_000  # To reach 1B total
        self.total_target = 1_000_000_000  # 1B

    def combine_gutenberg_parts(self):
        """Combine both Gutenberg download parts"""
        print("📚 Combining Gutenberg datasets...")

        gutenberg_texts = []

        # Part 1: Original download
        part1_dir = Path("./gutenberg_600m/raw_books")
        if part1_dir.exists():
            part1_files = list(part1_dir.glob("*.txt"))
            print(f"Found {len(part1_files)} books in part 1")
            gutenberg_texts.extend(self.process_gutenberg_files(part1_files, "Part 1"))

        # Part 2: Second download
        part2_dir = Path("./gutenberg_600m_part2/raw_books")
        if part2_dir.exists():
            part2_files = list(part2_dir.glob("*.txt"))
            print(f"Found {len(part2_files)} books in part 2")
            gutenberg_texts.extend(self.process_gutenberg_files(part2_files, "Part 2"))

        current_tokens = sum(t['word_count'] * 0.75 for t in gutenberg_texts)
        print(f"📊 Total Gutenberg tokens: {current_tokens:,.0f}")

        return gutenberg_texts

    def process_gutenberg_files(self, file_list, source_name):
        """Process a list of Gutenberg files"""
        processed_texts = []

        for book_file in tqdm(file_list, desc=f"Processing {source_name}"):
            try:
                with open(book_file, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()

                cleaned_text = self.clean_gutenberg_text(raw_text)

                if len(cleaned_text) > 1000:
                    processed_texts.append({
                        'text': cleaned_text,
                        'source': 'gutenberg',
                        'word_count': len(cleaned_text.split()),
                        'file_source': source_name
                    })

            except Exception as e:
                continue

        return processed_texts

    def clean_gutenberg_text(self, text):
        """Clean Project Gutenberg text"""
        import re

        lines = text.split('\n')
        start_idx = 0
        end_idx = len(lines)

        # Find content boundaries
        for i, line in enumerate(lines[:200]):
            if re.search(r'\*\*\* START OF (THE|THIS) PROJECT GUTENBERG', line.upper()):
                start_idx = i + 1
                break

        for i in range(len(lines) - 1, max(0, len(lines) - 200), -1):
            if re.search(r'\*\*\* END OF (THE|THIS) PROJECT GUTENBERG', lines[i].upper()):
                end_idx = i
                break

        content = '\n'.join(lines[start_idx:end_idx])
        content = re.sub(r'\r\n', '\n', content)
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = re.sub(r'_+', '', content)
        content = re.sub(r'\[Illustration[^\]]*\]', '', content)
        content = re.sub(r'\[Footnote[^\]]*\]', '', content)

        return content.strip()

    def load_conversational_datasets(self):
        """Load all the transferred conversational datasets"""
        print(f"💬 Loading conversational datasets...")

        conversational_texts = []
        current_tokens = 0
        hf_data_dir = Path("./hf_conversational_datasets")

        if not hf_data_dir.exists():
            print(f"❌ No conversational datasets found at {hf_data_dir}")
            return []

        # Load each dataset
        dataset_configs = [
            ("alpaca", self.process_alpaca),
            ("dolly", self.process_dolly),
            ("ultrachat", self.process_ultrachat),
            ("wizardlm", self.process_wizardlm),
            ("openorca", self.process_openorca),
        ]

        for dataset_name, process_func in dataset_configs:
            dataset_path = hf_data_dir / dataset_name

            if dataset_path.exists():
                print(f"📥 Loading {dataset_name}...")
                try:
                    dataset = load_from_disk(str(dataset_path))
                    print(f"Found {len(dataset):,} examples in {dataset_name}")

                    # Process the dataset
                    processed = process_func(dataset, current_tokens)
                    conversational_texts.extend(processed)

                    # Update token count
                    batch_tokens = sum(t['word_count'] * 0.75 for t in processed)
                    current_tokens += batch_tokens

                    print(f"✅ Added {len(processed):,} examples ({batch_tokens:,.0f} tokens)")

                    # Stop if we have enough tokens
                    if current_tokens >= self.conversational_target:
                        print(f"🎯 Conversational target reached: {current_tokens:,.0f} tokens")
                        break

                except Exception as e:
                    print(f"❌ Error loading {dataset_name}: {e}")
                    continue
            else:
                print(f"⚠️ {dataset_name} not found, skipping...")

        print(f"💬 Total conversational examples: {len(conversational_texts):,}")
        print(f"🔤 Total conversational tokens: {current_tokens:,.0f}")

        return conversational_texts

    def process_alpaca(self, dataset, current_tokens):
        """Process Alpaca dataset"""
        processed = []

        for example in tqdm(dataset, desc="Processing Alpaca"):
            if current_tokens >= self.conversational_target:
                break

            try:
                instruction = example.get('instruction', '')
                input_text = example.get('input', '')
                output = example.get('output', '')

                if input_text:
                    text = f"Human: {instruction}\n{input_text}\nAssistant: {output}"
                else:
                    text = f"Human: {instruction}\nAssistant: {output}"

                word_count = len(text.split())
                tokens = word_count * 0.75

                if current_tokens + tokens <= self.conversational_target:
                    processed.append({
                        'text': text,
                        'source': 'alpaca',
                        'word_count': word_count
                    })
                    current_tokens += tokens

            except Exception as e:
                continue

        return processed

    def process_dolly(self, dataset, current_tokens):
        """Process Dolly dataset"""
        processed = []

        for example in tqdm(dataset, desc="Processing Dolly"):
            if current_tokens >= self.conversational_target:
                break

            try:
                instruction = example.get('instruction', '')
                context = example.get('context', '')
                response = example.get('response', '')

                if context:
                    text = f"Human: {instruction}\nContext: {context}\nAssistant: {response}"
                else:
                    text = f"Human: {instruction}\nAssistant: {response}"

                word_count = len(text.split())
                tokens = word_count * 0.75

                if current_tokens + tokens <= self.conversational_target:
                    processed.append({
                        'text': text,
                        'source': 'dolly',
                        'word_count': word_count
                    })
                    current_tokens += tokens

            except Exception as e:
                continue

        return processed

    def process_ultrachat(self, dataset, current_tokens):
        """Process UltraChat dataset"""
        processed = []

        for example in tqdm(dataset, desc="Processing UltraChat"):
            if current_tokens >= self.conversational_target:
                break

            try:
                conversation = ""
                data = example.get('data', [])

                for turn in data:
                    if turn.get('role') == 'user':
                        conversation += f"Human: {turn.get('content', '')}\n"
                    elif turn.get('role') == 'assistant':
                        conversation += f"Assistant: {turn.get('content', '')}\n"

                if len(conversation) > 100:
                    word_count = len(conversation.split())
                    tokens = word_count * 0.75

                    if current_tokens + tokens <= self.conversational_target:
                        processed.append({
                            'text': conversation,
                            'source': 'ultrachat',
                            'word_count': word_count
                        })
                        current_tokens += tokens

            except Exception as e:
                continue

        return processed

    def process_wizardlm(self, dataset, current_tokens):
        """Process WizardLM dataset"""
        processed = []

        for example in tqdm(dataset, desc="Processing WizardLM"):
            if current_tokens >= self.conversational_target:
                break

            try:
                conversations = example.get('conversations', [])
                if len(conversations) >= 2:
                    human_msg = conversations[0].get('value', '')
                    assistant_msg = conversations[1].get('value', '')

                    text = f"Human: {human_msg}\nAssistant: {assistant_msg}"

                    word_count = len(text.split())
                    tokens = word_count * 0.75

                    if current_tokens + tokens <= self.conversational_target:
                        processed.append({
                            'text': text,
                            'source': 'wizardlm',
                            'word_count': word_count
                        })
                        current_tokens += tokens

            except Exception as e:
                continue

        return processed

    def process_openorca(self, dataset, current_tokens):
        """Process OpenOrca dataset"""
        processed = []

        for example in tqdm(dataset, desc="Processing OpenOrca"):
            if current_tokens >= self.conversational_target:
                break

            try:
                question = example.get('question', '')
                response = example.get('response', '')

                text = f"Human: {question}\nAssistant: {response}"

                word_count = len(text.split())
                tokens = word_count * 0.75

                if current_tokens + tokens <= self.conversational_target:
                    processed.append({
                        'text': text,
                        'source': 'openorca',
                        'word_count': word_count
                    })
                    current_tokens += tokens

            except Exception as e:
                continue

        return processed

    def create_billion_token_dataset(self):
        """Create the complete 1B token dataset"""
        print("🚀 BUILDING 1 BILLION TOKEN DATASET")
        print("=" * 50)

        # Get Gutenberg data
        gutenberg_texts = self.combine_gutenberg_parts()
        gutenberg_tokens = sum(t['word_count'] * 0.75 for t in gutenberg_texts)

        # Get conversational data
        conversational_texts = self.load_conversational_datasets()
        conversational_tokens = sum(t['word_count'] * 0.75 for t in conversational_texts)

        # Combine everything
        all_texts = gutenberg_texts + conversational_texts
        total_tokens = gutenberg_tokens + conversational_tokens

        print(f"\n🎯 FINAL DATASET COMPOSITION:")
        print(f"  📚 Gutenberg books: {len(gutenberg_texts):,}")
        print(f"  📚 Gutenberg tokens: {gutenberg_tokens:,.0f} ({gutenberg_tokens / total_tokens * 100:.1f}%)")
        print(f"  💬 Conversational examples: {len(conversational_texts):,}")
        print(
            f"  💬 Conversational tokens: {conversational_tokens:,.0f} ({conversational_tokens / total_tokens * 100:.1f}%)")
        print(f"  🎯 TOTAL TOKENS: {total_tokens:,.0f}")

        if total_tokens >= 800_000_000:  # At least 800M is great
            print("✅ EXCELLENT TOKEN COUNT ACHIEVED!")
        else:
            print(f"⚠️ Consider adding more data for optimal results")

        # Tokenize and save
        return self.tokenize_billion_dataset(all_texts)

    def tokenize_billion_dataset(self, texts, tokenizer_path="./hf_models/mistral-7b"):
        """Tokenize the massive dataset efficiently"""
        print(f"\n🔤 Tokenizing {len(texts):,} texts...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            return None

        # Create dataset in chunks to manage memory
        print("📦 Creating dataset in chunks...")
        chunk_size = 5000  # Smaller chunks for memory efficiency
        all_datasets = []

        for i in range(0, len(texts), chunk_size):
            chunk_texts = texts[i:i + chunk_size]
            print(f"Processing chunk {i // chunk_size + 1}/{(len(texts) - 1) // chunk_size + 1}")

            chunk_dict = {
                'text': [item['text'] for item in chunk_texts],
                'source': [item['source'] for item in chunk_texts],
                'word_count': [item['word_count'] for item in chunk_texts]
            }

            chunk_dataset = Dataset.from_dict(chunk_dict)

            # Tokenize chunk
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=1024,
                    return_tensors=None
                )

            tokenized_chunk = chunk_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=50,  # Smaller batch size
                remove_columns=['text']
            )

            all_datasets.append(tokenized_chunk)

            # Clean up memory
            del chunk_dict, chunk_dataset, tokenized_chunk
            gc.collect()

        # Combine all chunks
        print("🔗 Combining all chunks...")
        final_dataset = concatenate_datasets(all_datasets)

        # Shuffle the final dataset
        print("🔀 Shuffling dataset...")
        final_dataset = final_dataset.shuffle(seed=42)

        # Save dataset
        output_path = self.output_dir / "tokenized_dataset"
        print(f"💾 Saving to {output_path}...")
        final_dataset.save_to_disk(str(output_path))

        # Save metadata
        metadata = {
            'total_texts': len(texts),
            'gutenberg_texts': len([t for t in texts if t['source'] == 'gutenberg']),
            'conversational_texts': len([t for t in texts if t['source'] != 'gutenberg']),
            'total_tokens': sum(len(item['input_ids']) for item in final_dataset),
            'target_tokens': self.total_target,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Billion token dataset saved to: {output_path}")
        print(f"📊 Final token count: {metadata['total_tokens']:,}")

        return final_dataset


def main():
    print("🌟 1 BILLION TOKEN DATASET BUILDER")
    print("=" * 50)
    print("Combining Gutenberg literature + conversational data")
    print("Target: 1 billion tokens for world-class base adapter")
    print()

    builder = BillionTokenBuilder()
    dataset = builder.create_billion_token_dataset()

    if dataset:
        print("\n🎉 1 BILLION TOKEN DATASET COMPLETE!")
        print("💰 Ready for premium Lambda Labs training!")
        print("🚀 This will be a WORLD-CLASS base adapter!")
        print("\nNext steps:")
        print("1. Upload dataset to Lambda Labs")
        print("2. Train base adapter with optimal settings")
        print("3. Stack specialized adapters on top")
        print("4. Deploy your AI agent ecosystem!")


if __name__ == "__main__":
    main()