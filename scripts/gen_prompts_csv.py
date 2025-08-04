import csv

examples = [
    {"prompt": "Write a short poem about the sea.", "act": "The sea whispers in blue sighs, under endless skies."},
    {"prompt": "Explain the Pythagorean theorem.", "act": "In a right triangle, a² + b² = c²."},
    {"prompt": "What's the capital of France?", "act": "Paris."}
]

with open("prompts.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "act"])
    writer.writeheader()
    writer.writerows(examples)

print("✅ prompts.csv created.")
