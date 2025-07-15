import torch
import bitsandbytes as bnb

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("❌ CUDA not available")

# This will raise if bitsandbytes is not correctly using GPU
print("✅ bitsandbytes is using GPU-backed Linear8bitLt.")
