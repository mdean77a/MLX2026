from mlx_lm import generate
from typing import Optional


def generate_response(
    model, 
    tokenizer, 
    user_message: str, 
    model_id: str = None,
    prompt_cache=None,
    reasoning_level: str = "low",
    **kwargs
):
    """
    Generate a response to a user message using the MLX language model.
    Automatically handles different prompt formats based on model type.
    
    Args:
        model: The loaded MLX model
        tokenizer: The tokenizer for the model
        user_message: The user's input message
        model_id: The model identifier (used to determine prompt format)
        prompt_cache: Optional prompt cache for multi-turn conversations
        reasoning_level: For GPT-OSS models, set reasoning effort: "low", "medium", or "high"
                        (default: "low" - reduces verbose internal analysis)
        **kwargs: Additional arguments to pass to the generate function
                  (e.g., max_tokens, temperature, top_p, etc.)
    """
    # Detect if this is a GPT-OSS model that requires Harmony format
    is_gpt_oss = model_id and ("gpt-oss" in model_id.lower() or "oss-gpt" in model_id.lower())
    
    if is_gpt_oss:
        system_msg = f"Reasoning: {reasoning_level}"
        prompt = _format_harmony_prompt(user_message, system_msg)
    elif tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": user_message}]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        prompt = user_message
    
    print(f"User message: {user_message}\n")
    
    # Set better defaults for GPT-OSS models if not provided
    if is_gpt_oss:
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 2048  # Generous limit to allow model to complete reasoning
    
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        verbose=False, 
        prompt_cache=prompt_cache, 
        **kwargs
    )
    
    # If GPT-OSS, extract only the final channel response
    if is_gpt_oss:
        response = _extract_harmony_final(response)
    
    print(f"{response}\n")
    
    return response


def _extract_harmony_final(response: str) -> str:
    """
    Extract the final response from Harmony format output.
    Harmony models output reasoning in analysis/commentary channels,
    then the final answer in the 'final' channel.
    
    Args:
        response: The raw response from a Harmony-formatted model
    
    Returns:
        Just the final response content, without channel markers
    """
    # Look for the final channel marker
    if "<|channel|>final<|message|>" in response:
        # Extract everything after the final channel marker
        parts = response.split("<|channel|>final<|message|>")
        if len(parts) > 1:
            final_response = parts[-1]  # Get the last occurrence
            # Clean up any end markers
            final_response = final_response.split("<|end|>")[0]
            return final_response.strip()
    
    # If no final channel found but has analysis/commentary, indicate the issue
    if "<|channel|>analysis<|message|>" in response or "<|channel|>commentary<|message|>" in response:
        return "[Model did not complete response - try increasing max_tokens or adjusting prompt]"
    
    # If no channel markers at all, return the original response
    return response


def _format_harmony_prompt(user_message: str, system_message: Optional[str] = None) -> str:
    """
    Format a prompt using the Harmony format for GPT-OSS models.
    
    Args:
        user_message: The user's input message
        system_message: Optional system instructions
    
    Returns:
        Harmony-formatted prompt string
    """
    try:
        from openai_harmony import (
            HarmonyEncodingName,
            load_harmony_encoding,
            Conversation,
            Message,
            Role,
            SystemContent,
        )
        
        # Load the Harmony encoding for GPT-OSS
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        
        # Build the conversation
        messages = [Message.from_role_and_content(Role.SYSTEM, SystemContent.new())]
        
        # Add system message if provided
        if system_message:
            from openai_harmony import DeveloperContent
            messages.append(
                Message.from_role_and_content(
                    Role.DEVELOPER,
                    DeveloperContent.new().with_instructions(system_message)
                )
            )
        
        # Add user message
        messages.append(Message.from_role_and_content(Role.USER, user_message))
        
        # Create conversation
        convo = Conversation.from_messages(messages)
        
        # Render for completion and decode to string
        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        prompt = encoding.decode(prefill_ids)
        
        return prompt
        
    except ImportError:
        print("⚠️  Warning: openai-harmony not installed. Install with: pip install openai-harmony")
        print("Falling back to plain prompt format.\n")
        return user_message


def generate_response_with_system(
    model,
    tokenizer,
    user_message: str,
    system_message: str = None,
    model_id: str = None,
    prompt_cache=None,
    reasoning_level: str = "low",
    **kwargs
):
    """
    Generate a response with an optional system message.
    Useful for setting behavior, tone, or context.
    
    Args:
        model: The loaded MLX model
        tokenizer: The tokenizer for the model
        user_message: The user's input message
        system_message: System-level instructions for the model
        model_id: The model identifier (used to determine prompt format)
        prompt_cache: Optional prompt cache for multi-turn conversations
        reasoning_level: For GPT-OSS models, set reasoning effort: "low", "medium", or "high"
        **kwargs: Additional arguments to pass to the generate function
    """
    is_gpt_oss = model_id and ("gpt-oss" in model_id.lower() or "oss-gpt" in model_id.lower())
    
    if is_gpt_oss:
        # Combine reasoning level with custom system message if provided
        full_system_msg = f"Reasoning: {reasoning_level}"
        if system_message:
            full_system_msg = f"{full_system_msg}\n{system_message}"
        prompt = _format_harmony_prompt(user_message, full_system_msg)
    elif tokenizer.chat_template is not None:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        prompt = f"{system_message}\n\n{user_message}" if system_message else user_message
    
    print(f"User message: {user_message}\n")
    if system_message:
        print(f"System: {system_message}\n")
    
    # Set better defaults for GPT-OSS models if not provided
    if is_gpt_oss:
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 2048  # Generous limit to allow model to complete reasoning
    
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        verbose=False,
        prompt_cache=prompt_cache,
        **kwargs
    )
    
    # If GPT-OSS, extract only the final channel response
    if is_gpt_oss:
        response = _extract_harmony_final(response)
    
    print(f"{response}\n")
    
    return response
