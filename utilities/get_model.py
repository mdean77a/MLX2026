from enum import Enum
from mlx_lm import load
from typing import Tuple


class ModelType(Enum):
    """Enumeration of available models with their full MLX identifiers."""
    QWEN3_4B = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    GPT_120B = "mlx-community/oss-gpt-120b-4bit"  # Example - update with actual ID
    LLAMA_8B = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"  # Example
    MISTRAL_7B = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"  # Example
    
    @property
    def short_name(self) -> str:
        """Get the short, friendly name for the model."""
        return self.name.replace('_', '-')
    
    @property
    def cache_name(self) -> str:
        """Get the name to use for cache files."""
        return self.value.split("/")[-1]


# Alias mapping for easier access
MODEL_ALIASES = {
    "qwen3-4b": ModelType.QWEN3_4B,
    "qwen": ModelType.QWEN3_4B,
    "gpt-120b": ModelType.GPT_120B,
    "gpt": ModelType.GPT_120B,
    "llama-8b": ModelType.LLAMA_8B,
    "llama": ModelType.LLAMA_8B,
    "mistral-7b": ModelType.MISTRAL_7B,
    "mistral": ModelType.MISTRAL_7B,
}


def get_model(model: str | ModelType = ModelType.QWEN3_4B, verbose: bool = True) -> Tuple:
    """
    Load a model and tokenizer.
    
    Args:
        model: Either a ModelType enum value, a string alias (e.g., "qwen3-4b"), 
               or a full model ID string
        verbose: Whether to print loading message
    
    Returns:
        tuple: (model, tokenizer, model_id)
            - model: The loaded MLX model
            - tokenizer: The tokenizer for the model
            - model_id: The full model identifier string
    
    Examples:
        # Using enum
        model, tokenizer, model_id = get_model(ModelType.QWEN3_4B)
        
        # Using alias
        model, tokenizer, model_id = get_model("qwen")
        
        # Using full ID
        model, tokenizer, model_id = get_model("mlx-community/Custom-Model-4bit")
    """
    # Determine the model ID
    if isinstance(model, ModelType):
        model_id = model.value
    elif isinstance(model, str):
        # Check if it's an alias
        model_lower = model.lower()
        if model_lower in MODEL_ALIASES:
            model_id = MODEL_ALIASES[model_lower].value
        else:
            # Assume it's a full model ID
            model_id = model
    else:
        raise ValueError(f"Invalid model type: {type(model)}")
    
    if verbose:
        print(f"Loading model: {model_id}")
        print("(First time may take a while...)")
    
    loaded_model, loaded_tokenizer = load(model_id)
    
    if verbose:
        print("✓ Model loaded successfully!")
    
    return loaded_model, loaded_tokenizer, model_id


def list_available_models() -> None:
    """Print all available models with their aliases."""
    print("Available Models:")
    print("-" * 60)
    for model_type in ModelType:
        print(f"\n{model_type.short_name}:")
        print(f"  Full ID: {model_type.value}")
        aliases = [alias for alias, m_type in MODEL_ALIASES.items() if m_type == model_type]
        print(f"  Aliases: {', '.join(aliases)}")