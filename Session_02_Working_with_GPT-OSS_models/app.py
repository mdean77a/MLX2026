from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache
from utilities import get_model, create_cache, generate_response, generate_response_with_system, list_available_models

model, tokenizer, MODEL_ID = get_model("llama")

# list_available_models()

generate_response(model, tokenizer, "What is the capital of the United States?", model_id=MODEL_ID)
generate_response(model, tokenizer, "What is the capital of the United Moons?", model_id=MODEL_ID)
generate_response(model, tokenizer, "Hi my name is Mike Dean.", model_id=MODEL_ID)
generate_response(model, tokenizer, "What is my name?", model_id=MODEL_ID)

prompt_cache, cache_file = create_cache(model, MODEL_ID)
# print(cache_file)
generate_response(model, tokenizer, "Hi my name is Mike Dean.", model_id=MODEL_ID, prompt_cache=prompt_cache)
generate_response(model, tokenizer, "What's my name?", model_id=MODEL_ID, prompt_cache=prompt_cache)
generate_response(model, tokenizer, "What's your name?", model_id=MODEL_ID, prompt_cache=prompt_cache)
generate_response(model, tokenizer, "Can you give me some advice about cooking rice?", model_id=MODEL_ID, prompt_cache=prompt_cache)

# Save the prompt cache to disk to reuse it at a later time
save_prompt_cache(cache_file, prompt_cache)

# Make a new prompt cache but don't save it or it will overwrite what we just saved
prompt_cache, _ = create_cache(model, MODEL_ID)
# print(cache_file)
generate_response(model, tokenizer, "What's my name?", model_id=MODEL_ID, prompt_cache=prompt_cache)

prompt_cache = load_prompt_cache(cache_file)

generate_response(model, tokenizer, "What's my name?", model_id=MODEL_ID, prompt_cache=prompt_cache)

generate_response(model, tokenizer, "Summarize what we have discussed, but do not repeat everything.", model_id=MODEL_ID, prompt_cache=prompt_cache)
generate_response(model, tokenizer, "Tell me the recipe again. Don't summarize it - I want the original version.", model_id=MODEL_ID, prompt_cache=prompt_cache, max_tokens=2048)

