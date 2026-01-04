from mlx_lm import load, generate
# Load the model
from mlx_lm import load
MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit" 
print("Loading model... (first time may take a while)")
model, tokenizer = load(MODEL_ID)
prompt = "What is the capital of the United Moons?"
if tokenizer.chat_template is not None:
    messages = [{"role":"user", "content":prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=False,
    )
response = generate(model, tokenizer, prompt=prompt, verbose=True)