# MLX2026
Explorations of using Apple MLX library to duplicate patterns used in AI Makerspace Bootcamps.

My goal is to use notebooks as executable "README.md" files that will also tell the user how to use the code that is outside of the notebooks.

Assumption for user is that they are using Apple silicon since MLX will not work elsewhere.

## Setup Notebook
This notebook will show how to setup the environment for the MLX library.  It will also include several examples of how to use the library.  In the notebook, we will explore loading the model, generating responses, and using prompt caching.  It will include multiple chats with and without continuity between turns.  Finally, it will show how to make the first steps at migrating the code to an application structure outside of a notebook.

### Setup Script
The last cell of the setup.ipynb was copied into setup.py.  If you execute this script in a terminalit will execute the same code as the notebook.  It is important to note that the script must be executed in the same environment as the notebook.  If you are using a virtual environment, you must activate it before executing the script.  If you use Cursor or VS Code and follow the instructions in the notebook, you will be able to execute the script from the terminal integrated within your editor.

If you copy setup.py into a fresh directory, you should also copy pyproject.toml into that directory and type `uv sync` to install the dependencies.  Then you should be able to execute the script from the terminal by typing `uv run python setup.py`.

## Refactor Notebook
This notebook will start with the code from the setup.py script and refactor it into a more modular structure. The ultimate result of this refactoring will be put into a single script that can be run from the command line.

### Refactor Script
This script is the refactored version of the setup.py script using the modulare structure described in the notebook.