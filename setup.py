from mlx_lm import load, generate
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
from pathlib import Path

MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit" 
print("Loading model... (first time may take a while)")
model, tokenizer = load(MODEL_ID)

response = generate(model, tokenizer, "What is the capital of the United States?")
print(response)

prompt = "What is the capital of the United States?"
if tokenizer.chat_template is not None:
    messages = [{"role":"user", "content":prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=False,
    )
response = generate(model, tokenizer, prompt=prompt, verbose=True)

prompt = "What is the capital of the United Moons?"
if tokenizer.chat_template is not None:
    messages = [{"role":"user", "content":prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=False,
    )
response = generate(model, tokenizer, prompt=prompt, verbose=True)

# User turn one
user_message = "Hi my name is Mike Dean."
print(f"First user message: {user_message}\n")
messages = [{"role": "user", "content": user_message}]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)

# Assistant response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=False,
)

print(f"First prompt response: {response}\n")

# User turn two
user_message = "What's my name?"
messages = [{"role": "user", "content": user_message}]
print(f"Second user message: {user_message}\n")
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize = False,
)

# Assistant response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=False,
)
print(f"Second prompt response: {response}\n")

"""
An example of a multi-turn chat with prompt caching.
"""

from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
from pathlib import Path

# MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit" 
# model, tokenizer = load(MODEL_ID)

# Make the initial prompt cache for the model
prompt_cache = make_prompt_cache(model)

# Create the cache files directory 
cache_dir = Path("cache_files")
cache_dir.mkdir(exist_ok=True)
model_name = MODEL_ID.split("/")[-1]
cache_file = cache_dir/f"{model_name}.safetensors"

# User turn
prompt = "Hi my name is Mike Dean."
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

# Assistant response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    prompt_cache=prompt_cache,
)

# User turn
prompt = "What's my name?"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

# Assistant response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    prompt_cache=prompt_cache,
)

# User turn
prompt = "What's your name?"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

# Assistant response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    prompt_cache=prompt_cache,
)

# User turn
prompt = "Can you give me some advice about cooking rice?"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

# Assistant response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    prompt_cache=prompt_cache,
)
# Save the prompt cache to disk to reuse it at a later time
save_prompt_cache(cache_file, prompt_cache)

# Load the prompt cache from disk
prompt_cache = load_prompt_cache(cache_file)

# User turn
prompt = "Summarize what we have discussed, but do not repeat everything."
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

# Assistant response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    prompt_cache=prompt_cache,
)

# User turn
prompt = "Tell me the recipe again. Don't summarize it - I want the original version."
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

# Assistant response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    max_tokens=2048,
    prompt_cache=prompt_cache,
)