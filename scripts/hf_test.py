from huggingface_hub import HfApi
api = HfApi()
print(api.whoami())   # Should print your user info dict

