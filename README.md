# MLX2026 - Using Apple silicon with MLX library

Explorations of using Apple MLX library to duplicate patterns used in AI Makerspace Bootcamps. This repo contains a series of session folders.  When you have cloned this repo to your local machine, you should open the session folder that you want to work on using Cursor or VS Code.  The README.md file in the session folder will contain the instructions for the session.

It is important that the dependencies are different for each session.  After you have opened the session folder with Cursor or VS Code, you should open an integrated terminal and run the `uv sync` command to install the dependencies for the session.  This will result in a .venv folder as well as a uv.lock file.  This means that there will be a different environment for each session folder!

Each session folder should be able to be used independently of the other session folders.  This means that I have often copied contents of a previous session folder.  The goal is to avoid you from having to find code that you need from previous sessions.

## 📁 Repository Structure

```text
MLX2026/
├── LICENSE
├── README.md
│
├── Session_01_Introduction_and_Refactoring/
│   ├── README.md                    # Session documentation
│   ├── pyproject.toml               # Dependencies (mlx-lm, ipykernel)
│   ├── uv.lock                      # Locked dependency versions
│   ├── setup.ipynb                  # Environment setup & MLX basics
│   ├── setup.py                     # Script version of setup notebook
│   ├── refactor.ipynb               # Code refactoring tutorial
│   ├── refactor.py                  # Refactored modular application
│   ├── cache_files/                 # Cached model files (.safetensors)
│   │   ├── Qwen3-4B-Instruct-2507-4bit.safetensors
│   │   └── Meta-Llama-3.1-8B-Instruct-8bit.safetensors
│   └── utilities/                   # Reusable utility modules
│       ├── __init__.py              # Package initialization
│       ├── get_model.py             # Model loading (Qwen, GPT-OSS, Llama, Mistral)
│       ├── utils.py                 # Response generation & Harmony format
│       └── create_cache.py          # Prompt caching utilities
│
├── Session_02_Working_with_GPT-OSS_models/
│   ├── README.md                    # Session documentation
│   ├── pyproject.toml               # Dependencies
│   ├── app.py                       # GPT-OSS application
│   └── utilities/                   # Reusable utility modules
│       ├── __init__.py              # Package initialization
│       ├── get_model.py             # Model loading utilities
│       ├── utils.py                 # Response generation
│       └── create_cache.py          # Prompt caching
│
├── Session_03_Creating_Simple_Web_Application/
│   └── (planned)
│
└── Session_04_Fine_Tuning_Models/
    └── (planned)
```
