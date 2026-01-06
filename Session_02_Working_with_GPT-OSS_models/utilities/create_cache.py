from mlx_lm.models.cache import make_prompt_cache
from pathlib import Path


def create_cache(model, model_id, cache_dir_name="cache_files"):
    """
    Create a prompt cache for the model and set up the cache directory.
    
    Args:
        model: The loaded MLX model
        model_id: The model identifier (e.g., "mlx-community/Qwen3-4B-Instruct-2507-4bit")
        cache_dir_name: Name of the directory to store cache files (default: "cache_files")
    
    Returns:
        tuple: (prompt_cache, cache_file_path)
            - prompt_cache: The initialized prompt cache object
            - cache_file_path: Path object pointing to the cache file
    """
    # Make the initial prompt cache for the model
    prompt_cache = make_prompt_cache(model)
    
    # Create the cache files directory 
    cache_dir = Path(cache_dir_name)
    cache_dir.mkdir(exist_ok=True)
    
    # Generate cache file path
    model_name = model_id.split("/")[-1]
    cache_file = cache_dir / f"{model_name}.safetensors"
    
    return prompt_cache, cache_file