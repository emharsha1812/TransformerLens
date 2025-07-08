import torch
from transformer_lens import HookedTransformer

# The Hugging Face model name you added
MODEL_NAME = "ibm-granite/granite-3.3-2b-instruct" 

print(f"--- Loading model: {MODEL_NAME} ---")

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the model from Hugging Face
# This will use all the code you've just written!
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=device,
    # You might need to use a lower-precision dtype if you're low on VRAM
    # torch_dtype=torch.bfloat16 
)

print("\n--- Model Configuration ---")
# Print the config to double-check that your parameters were loaded correctly
print(model.cfg)

print("\n--- Running Generation Test ---")
prompt = "The best programming language is"

# Generate some text
# The model will return token IDs, so we use .to_string() to convert back to text
output = model.generate(
    prompt,
    max_new_tokens=10,
    temperature=0.7,
)

print(f"\nPrompt: '{prompt}'")
print(f"Generated text: '{output}'")

print("\n--- ✅ Test Complete ---")