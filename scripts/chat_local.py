# scripts/chat_local.py

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "qlora_outputs/base_gutenberg_2ep"
tokenizer_dir = "C:/Users/pkidw/PycharmProjects/hf_models/mistral-7b"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
model_base = AutoModelForCausalLM.from_pretrained(
    tokenizer_dir,
    load_in_4bit=True,
    device_map="auto",
    local_files_only=True,
)
model = PeftModel.from_pretrained(model_base, model_dir, local_files_only=True)
model.eval()

def respond(prompt, max_len=256, temp=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_len, temperature=temp)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(
    fn=respond,
    inputs=["text", gr.Slider(50,500,value=256,label="Max new tokens"),
            gr.Slider(0.1,1.0,step=0.1,value=0.7,label="Temperature")],
    outputs="text",
    title="Local Mistral-LoRA Chat",
    description="Chat with your Gutenberg-base LoRA-enhanced model locally."
)
iface.launch(server_name="0.0.0.0", share=False)
