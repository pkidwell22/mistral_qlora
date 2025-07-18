# scripts/chat_gradio.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr

# ─── Load Base + LoRA Adapter ──────────────────────────────────────────
base_model = "/mnt/c/Users/pkidw/hf_models/mistral-7b"
adapter_path = "adapters/hle_dolly_adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# ─── Chat Function ─────────────────────────────────────────────────────
def chat_fn(message, history):
    # Build prompt from limited recent history
    max_turns = 4  # limit history to last 4 exchanges (8 entries)
    trimmed_history = history[-max_turns * 2:] if history else []

    prompt = ""
    for turn in trimmed_history:
        if turn["role"] == "user":
            prompt += f"### Instruction:\n{turn['content']}\n\n"
        elif turn["role"] == "assistant":
            prompt += f"### Response:\n{turn['content']}\n\n"

    prompt += f"### Instruction:\n{message}\n\n### Response:\n"

    # Tokenize input and truncate if needed
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    input_length = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded_output = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
    response = decoded_output.strip()

    # Update chat history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history


# ─── Gradio Interface ──────────────────────────────────────────────────
chat_ui = gr.ChatInterface(
    fn=chat_fn,
    chatbot=gr.Chatbot(type="messages"),
    title="🧠 Mistral Chat",
    examples=[
        "What are some philosophical insights from art?",
        "Explain how abstraction can convey truth better than realism.",
        "How does storytelling shape collective memory?",
    ],
)

chat_ui.launch(server_name="0.0.0.0", server_port=7863)
