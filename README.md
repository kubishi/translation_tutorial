# Yaduha translation tutorial

An interactive Jupyter notebook that walks through the **LLM-Assisted Rule-Based Machine Translation (LLM-RBMT)** approach we used to build an English → Owens Valley Paiute translator.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=1242685188)

## Setup

Requires [uv](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync                           # install yaduha, yaduha-ovp, jupyter, etc.
uv run jupyter lab tutorial.ipynb # or: uv run jupyter notebook
```

If you use VS Code, open the folder, open `tutorial.ipynb`, and pick the `.venv` kernel that `uv sync` created.

## API key

Several cells call the OpenAI API. To run them, drop a `.env` file next to the notebook:

```
OPENAI_API_KEY=sk-...
```
