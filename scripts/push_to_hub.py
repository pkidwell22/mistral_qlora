# scripts/push_to_hub.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    load_in_8bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(base, "qlora_mistral_output/final")
tokenizer = AutoTokenizer.from_pretrained("qlora_mistral_output/final", trust_remote_code=True)

model.push_to_hub("pkidwell22/mistral-alpaca-qlora")
tokenizer.push_to_hub("pkidwell22/mistral-alpaca-qlora")
print("✅ Pushed to Hugging Face Hub")
