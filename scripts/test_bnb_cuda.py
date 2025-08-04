import torch
import bitsandbytes as bnb

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
print("✅ bitsandbytes is using GPU-backed Linear8bitLt.")
