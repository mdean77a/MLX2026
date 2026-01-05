from mlx_lm import generate


def generate_response(model, tokenizer, user_message, prompt_cache=None, **kwargs):
    """
    Generate a response to a user message using the MLX language model.
    
    Args:
        model: The loaded MLX model
        tokenizer: The tokenizer for the model
        user_message: The user's input message
        prompt_cache: Optional prompt cache for multi-turn conversations
        **kwargs: Additional arguments to pass to the generate function
                  (e.g., max_tokens, temperature, top_p, etc.)
    """
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": user_message}]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        prompt = user_message
    
    print(f"User message: {user_message}\n")
    print(f"{generate(model, tokenizer, prompt=prompt, verbose=False, prompt_cache=prompt_cache, **kwargs)}\n")