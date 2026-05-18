# Yaduha translation tutorial

An interactive Jupyter notebook that walks through the **LLM-Assisted Rule-Based Machine Translation (LLM-RBMT)** approach we used to build an English → Owens Valley Paiute translator.

The notebook is written for two audiences at once: linguists / language-community members (no programming background assumed) and ML researchers / engineers. It mixes background, live LLM calls that fail, a tour of the OVP grammar encoded as Pydantic models, and a side-by-side comparison of structured vs. prompt-only translation.

Open [tutorial.ipynb](tutorial.ipynb).

## Setup

Requires [uv](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync                           # install yaduha, yaduha-ovp, jupyter, etc.
uv run jupyter lab tutorial.ipynb # or: uv run jupyter notebook
```

If you use VS Code, open the folder, open `tutorial.ipynb`, and pick the `.venv` kernel that `uv sync` created.

## API key (optional)

Several cells call the OpenAI API. To run them, drop a `.env` file next to the notebook:

```
OPENAI_API_KEY=sk-...
```

Cells that need a key are gated &mdash; the rest of the notebook (grammar, Pydantic models, deterministic rendering, interactive widgets) runs without one.

## Layout

- `yaduha/` &mdash; the framework (translators, agents, evaluators, language loader).
- `yaduha-ovp/` &mdash; the Owens Valley Paiute language pack (vocabulary, grammar models, prompts).
- `paper_llm_rbmt/` &mdash; the original LLM-RBMT paper (introducing the Pipeline Translator).
- `paper_llm_rbmt_2/` &mdash; the systematic five-translator evaluation paper.
- `powerpoints/` &mdash; talk decks the notebook narrative draws from.
- `build_notebook.py` &mdash; regenerates `tutorial.ipynb` from a structured cell list. Edit there if you want to change the notebook content.
