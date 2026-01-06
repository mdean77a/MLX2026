
# Session 2.  Working with GPT-OSS Models

The prompt formatting that MLX uses by default is not compatible with GPT-OSS models.  The OpenAI Harmony library provides a way to work with GPT-OSS models.  This library is a wrapper around the OpenAI API and provides a way to work with the models in a more structured way.  It is important to note that the OpenAI Harmony library is not yet fully supported by the MLX library.  We will be using the OpenAI Harmony library to work with the GPT-OSS models.

The openai-harmony library has been added to the dependencies in the pyproject.toml file.  Run `uv sync` to install the dependencies.