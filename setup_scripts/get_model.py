# Load the model
from mlx_lm import load
MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit" 
print("Loading model... (first time may take a while)")
model, tokenizer = load(MODEL_ID)