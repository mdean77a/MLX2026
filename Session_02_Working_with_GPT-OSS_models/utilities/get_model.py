from enum import Enum
from mlx_lm import load
from typing import Tuple, Optional
import os


class ModelType(Enum):
    """Enumeration of available models with their full MLX identifiers."""
    QWEN3_4B = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    GPT_120B = "mlx-community/gpt-oss-120b-MXFP4-Q4"  
    GPT_20B = "mlx-community/gpt-oss-20b-MXFP4-Q4" 
    LLAMA_8B = "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit" 
    MISTRAL_24B = "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit"
    
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
    "gpt-20b": ModelType.GPT_20B,
    "gpt": ModelType.GPT_20B,
    "llama-8b": ModelType.LLAMA_8B,
    "llama": ModelType.LLAMA_8B,
    "mistral": ModelType.MISTRAL_24B,
}


def get_model(
    model: str | ModelType = ModelType.QWEN3_4B, 
    verbose: bool = True,
    hf_token: Optional[str] = None
) -> Tuple:
    """
    Load a model and tokenizer.
    
    Args:
        model: Either a ModelType enum value, a string alias (e.g., "qwen3-4b"), 
               or a full model ID string
        verbose: Whether to print loading message
        hf_token: Hugging Face token for gated models. If None, will try to get 
                  from HF_TOKEN environment variable
    
    Returns:
        tuple: (model, tokenizer, model_id)
            - model: The loaded MLX model
            - tokenizer: The tokenizer for the model
            - model_id: The full model identifier string
    
    Examples:
        # Using enum
        model, tokenizer, model_id = get_model(ModelType.QWEN3_4B)
        
        # Using alias with token
        model, tokenizer, model_id = get_model("gpt-120b", hf_token="hf_...")
        
        # Token from environment
        model, tokenizer, model_id = get_model("gpt-120b")
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
    
    # Get token from parameter or environment
    token = hf_token or os.getenv("HF_TOKEN")
    
    if verbose:
        print(f"Loading model: {model_id}")
        if token:
            print("🔑 Using authentication token")
        print("(First time may take a while...)")
    
    try:
        # Pass token to load function if available
        if token:
            loaded_model, loaded_tokenizer = load(model_id, tokenizer_config={"token": token})
        else:
            loaded_model, loaded_tokenizer = load(model_id)
        
        if verbose:
            print("✓ Model loaded successfully!")
        
        return loaded_model, loaded_tokenizer, model_id
    
    except Exception as e:
        if "401" in str(e) or "authorization" in str(e).lower():
            print("\n❌ Authorization Error!")
            print("This model requires authentication. Please:")
            print("1. Get a token from https://huggingface.co/settings/tokens")
            print("2. Set it as environment variable: export HF_TOKEN='your_token'")
            print("   OR pass it directly: get_model('gpt-120b', hf_token='your_token')")
        raise


def list_available_models() -> None:
    """Print all available models with their aliases."""
    print("Available Models:")
    print("-" * 60)
    for model_type in ModelType:
        print(f"\n{model_type.short_name}:")
        print(f"  Full ID: {model_type.value}")
        aliases = [alias for alias, m_type in MODEL_ALIASES.items() if m_type == model_type]
        print(f"  Aliases: {', '.join(aliases)}")