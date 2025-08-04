#!/usr/bin/env python3
"""
HuggingFace Dataset Downloader for PyCharm (Windows)
Download conversational datasets with auth, then transfer to Ubuntu
"""

import os
from pathlib import Path
from datasets import load_dataset
import json
from tqdm import tqdm


class HFDatasetDownloader:
    def __init__(self, output_dir="./hf_conversational_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Target: 400M conversational tokens
        self.target_tokens = 400_000_000
        self.current_tokens = 0

    def download_sharegpt(self):
        """Download ShareGPT conversations"""
        print("📥 Downloading ShareGPT...")

        try:
            # Download full dataset
            dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")

            # Save raw dataset
            output_path = self.output_dir / "sharegpt"
            dataset.save_to_disk(str(output_path))

            print(f"✅ ShareGPT saved: {len(dataset):,} examples")
            print(f"📁 Location: {output_path}")

            return len(dataset)

        except Exception as e:
            print(f"❌ ShareGPT failed: {e}")
            return 0

    def download_ultrachat(self):
        """Download UltraChat (requires HF auth)"""
        print("📥 Downloading UltraChat...")

        try:
            # Download in chunks to manage size
            dataset = load_dataset("stingning/ultrachat", split="train")

            output_path = self.output_dir / "ultrachat"
            dataset.save_to_disk(str(output_path))

            print(f"✅ UltraChat saved: {len(dataset):,} examples")
            print(f"📁 Location: {output_path}")

            return len(dataset)

        except Exception as e:
            print(f"❌ UltraChat failed (likely needs HF auth): {e}")
            print("💡 Run: huggingface-cli login")
            return 0

    def download_openorca(self):
        """Download OpenOrca dataset"""
        print("📥 Downloading OpenOrca...")

        try:
            dataset = load_dataset("Open-Orca/OpenOrca", split="train")

            output_path = self.output_dir / "openorca"
            dataset.save_to_disk(str(output_path))

            print(f"✅ OpenOrca saved: {len(dataset):,} examples")
            print(f"📁 Location: {output_path}")

            return len(dataset)

        except Exception as e:
            print(f"❌ OpenOrca failed: {e}")
            return 0

    def download_alpaca(self):
        """Download Alpaca dataset"""
        print("📥 Downloading Alpaca...")

        try:
            dataset = load_dataset("tatsu-lab/alpaca", split="train")

            output_path = self.output_dir / "alpaca"
            dataset.save_to_disk(str(output_path))

            print(f"✅ Alpaca saved: {len(dataset):,} examples")
            print(f"📁 Location: {output_path}")

            return len(dataset)

        except Exception as e:
            print(f"❌ Alpaca failed: {e}")
            return 0

    def download_wizardlm(self):
        """Download WizardLM dataset"""
        print("📥 Downloading WizardLM...")

        try:
            dataset = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train")

            output_path = self.output_dir / "wizardlm"
            dataset.save_to_disk(str(output_path))

            print(f"✅ WizardLM saved: {len(dataset):,} examples")
            print(f"📁 Location: {output_path}")

            return len(dataset)

        except Exception as e:
            print(f"❌ WizardLM failed: {e}")
            return 0

    def download_dolly(self):
        """Download Dolly dataset"""
        print("📥 Downloading Dolly...")

        try:
            dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

            output_path = self.output_dir / "dolly"
            dataset.save_to_disk(str(output_path))

            print(f"✅ Dolly saved: {len(dataset):,} examples")
            print(f"📁 Location: {output_path}")

            return len(dataset)

        except Exception as e:
            print(f"❌ Dolly failed: {e}")
            return 0

    def create_transfer_package(self):
        """Create a summary for transfer to Ubuntu"""
        summary = {
            "download_location": str(self.output_dir.absolute()),
            "datasets_downloaded": [],
            "total_examples": 0,
            "estimated_tokens": 0,
            "transfer_instructions": [
                "1. Zip the entire hf_conversational_datasets folder",
                "2. Transfer to Ubuntu via WSL file system or SCP",
                "3. Run the billion_token_builder.py script",
                "4. Datasets will be automatically processed and combined"
            ]
        }

        # Check what was actually downloaded
        for dataset_dir in self.output_dir.iterdir():
            if dataset_dir.is_dir():
                try:
                    # Try to load and count
                    dataset = load_dataset(str(dataset_dir), split="train")
                    examples = len(dataset)
                    estimated_tokens = examples * 150  # Rough estimate

                    summary["datasets_downloaded"].append({
                        "name": dataset_dir.name,
                        "examples": examples,
                        "estimated_tokens": estimated_tokens,
                        "path": str(dataset_dir)
                    })

                    summary["total_examples"] += examples
                    summary["estimated_tokens"] += estimated_tokens

                except Exception as e:
                    continue

        # Save summary
        with open(self.output_dir / "download_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def download_all_datasets(self):
        """Download all available conversational datasets"""
        print("🚀 DOWNLOADING CONVERSATIONAL DATASETS")
        print("=" * 50)
        print("Target: 400M tokens for hybrid dataset")
        print()

        # Download order (most reliable first)
        download_functions = [
            self.download_alpaca,  # Most reliable, no auth
            self.download_dolly,  # Reliable, no auth
            self.download_sharegpt,  # Large, usually works
            self.download_wizardlm,  # Good quality
            self.download_openorca,  # Large instruction dataset
            self.download_ultrachat,  # Requires auth, try last
        ]

        total_examples = 0

        for download_func in download_functions:
            try:
                examples = download_func()
                total_examples += examples
                print(f"📊 Running total: {total_examples:,} examples")
                print()
            except Exception as e:
                print(f"❌ {download_func.__name__} failed: {e}")
                print()

        # Create transfer package
        summary = self.create_transfer_package()

        print("🎉 DOWNLOAD PHASE COMPLETE!")
        print("=" * 40)
        print(f"📊 Total datasets: {len(summary['datasets_downloaded'])}")
        print(f"📊 Total examples: {summary['total_examples']:,}")
        print(f"🔤 Estimated tokens: {summary['estimated_tokens']:,}")
        print(f"📁 Location: {self.output_dir}")
        print()
        print("🔄 NEXT STEPS:")
        print("1. Zip the hf_conversational_datasets folder")
        print("2. Transfer to Ubuntu (WSL or SCP)")
        print("3. Run billion_token_builder.py")
        print("4. Upload to Lambda Labs for training!")


# Windows PyCharm execution
if __name__ == "__main__":
    # First, login to HuggingFace
    print("🔐 HUGGINGFACE AUTHENTICATION")
    print("Run this first: huggingface-cli login")
    print("Enter your HF token when prompted")
    print()

    input("Press Enter when you've completed HF login...")

    # Download datasets
    downloader = HFDatasetDownloader()
    downloader.download_all_datasets()