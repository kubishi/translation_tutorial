"""Build tutorial.ipynb from structured cell definitions.

Run with: uv run python build_notebook.py
"""

import json
from pathlib import Path

cells: list[dict] = []


def md(text: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
    )


def code(text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }
    )


# ============================================================================
# Title + audience
# ============================================================================
md(
    """# Translating an Endangered Language with LLMs

### *A hands-on tour of the Yaduha framework and the LLM-RBMT paradigm*

This notebook is a guided tour of the research that built an English → **Owens Valley Paiute** translator. It is written for two audiences at once:

- **Linguists, language teachers, and community members**: you do not need to be a programmer. Read the explanations and run the cells with **Shift + Enter** &mdash; the code is short and labeled. You will see *why* general-purpose AI translators fail for endangered languages, and *how* the approach in this notebook fixes that.
- **Programmers and ML researchers**: every claim is backed by runnable code that uses the actual `yaduha` and `yaduha-ovp` packages. You can fork any cell and experiment.

By the end you will have:

1. Tried to translate English into Owens Valley Paiute with a state-of-the-art LLM directly &mdash; and seen it produce nonsense.
2. Understood the grammar of Owens Valley Paiute well enough to recognize a correct sentence.
3. Watched a Pydantic data model **act as a grammar** &mdash; building grammatically correct sentences by construction.
4. Run the full **LLM-RBMT pipeline**: an LLM that *uses* the grammar instead of inventing one.
5. Run a small experiment comparing the naive approach with the structured approach on the same inputs."""
)

# ============================================================================
# Setup
# ============================================================================
md(
    """## 0. Setup

This notebook expects two local Python packages to be installed (the framework `yaduha` and the language pack `yaduha-ovp`). If you opened this notebook from inside `kubishi/translation/`, that has already been done for you with [uv](https://docs.astral.sh/uv/).

If you want to set up the environment yourself:

```bash
# from the kubishi/translation/ directory
uv sync
uv run jupyter lab tutorial.ipynb
```

To run the LLM cells you need an **OpenAI API key**. If you do not have one, you can still run the deterministic cells &mdash; we will mark which is which.

Put your key in a `.env` file next to this notebook:

```
OPENAI_API_KEY=sk-...
```

Then run the cell below."""
)

code(
    """import os
from dotenv import load_dotenv

load_dotenv()  # reads .env from this directory

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pre-declare the LLM-backed objects so downstream cells degrade gracefully
# whether or not you have a key, and whether or not you ran the cells in order.
agent = None
pipeline = None
instructions = None

if OPENAI_API_KEY:
    print("OpenAI key loaded — LLM cells will run.")
else:
    print("No OPENAI_API_KEY found — LLM cells will be skipped, but everything else still works.")
"""
)

# ============================================================================
# Background on OVP
# ============================================================================
md(
    """## 1. Background: Owens Valley Paiute

**Owens Valley Paiute** (OVP, also *Eastern Mono* or *Monache*; ISO code `mnr`) is an Indigenous language of the Numic group of the Uto-Aztecan family, spoken in the Owens Valley region of eastern California.

Some facts that shape the rest of this notebook:

- It is **critically endangered**: 37 to 41 fluent speakers were reported in 1994, and the number is smaller now.
- There is **no publicly available parallel corpus** of OVP and English &mdash; nothing like the millions of aligned sentences that train Google Translate.
- Mistranslations are not just embarrassing &mdash; they can spread through learner materials and *erode* the language being revitalized.

This is what researchers call an **extremely low-resource** (or "no-resource") language. Almost every familiar machine translation technique &mdash; statistical MT, neural MT, fine-tuning a large model &mdash; assumes you have at least a few thousand aligned sentences. We have none.

So the question is: **can we translate into OVP at all, using only what is available &mdash; a dictionary, a grammar description, and a handful of example sentences?**"""
)

# ============================================================================
# Naive LLM baseline
# ============================================================================
md(
    """## 2. The naive baseline: just ask an LLM

Modern LLMs translate dozens of high-resource languages almost perfectly. A reasonable first instinct is: *let me just ask GPT to translate English into Owens Valley Paiute.*

Let's try it. The cell below sends a few English sentences to `gpt-4o` with no special prompting and shows what comes back.

> **If you do not have an API key**, skip this cell &mdash; we have included representative outputs in the markdown right after, so you can still follow along."""
)

code(
    """from openai import OpenAI

NAIVE_SENTENCES = [
    "I see the dog.",
    "The coyote ran.",
    "You are eating the apple.",
    "We are laughing.",
]

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
    for english in NAIVE_SENTENCES:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a translator from English to Owens Valley Paiute (also called Eastern Mono, ISO code mnr). Respond with ONLY the translation."},
                {"role": "user", "content": english},
            ],
        )
        print(f"EN:  {english}")
        print(f"OVP: {resp.choices[0].message.content.strip()}\\n")
else:
    print("(no API key — see the discussion below for representative output)")
"""
)

md(
    """### What just happened?

If you ran the cell, you saw output that *looks* like a translation. It usually is not.

The model has seen Wikipedia pages and sparse linguistic notes that mention Paiute, but it has never seen aligned English/OVP sentences during training. So it does what LLMs always do under those conditions: it **hallucinates fluently**. The output is in a script that looks vaguely Numic, the words have plausible shapes &mdash; and almost none of it is grammatical.

A native or intermediate speaker reviewing this output would tell you: the verbs are missing tense suffixes, the subject and object suffixes are wrong or swapped, the morphology that distinguishes "this dog" from "that dog" is missing, and several "words" are simply invented.

For a critically endangered language, **fluent-looking nonsense is worse than nothing**. It pollutes learning materials with plausible-sounding errors that are very hard to catch without a fluent speaker, and fluent speakers are exactly what we do not have.

So we need a different approach. Before we can describe it, we need to look at OVP grammar."""
)

# ============================================================================
# Grammar
# ============================================================================
md(
    """## 3. A whirlwind tour of OVP grammar

This section is the part of a typical talk that sounds like "linguistics" instead of "machine learning". Stick with it &mdash; the next section is going to mirror it directly in code.

### 3.1 Word order

OVP is a **Subject–Object–Verb (SOV)** language. The verb sits at the end of the sentence:

| English (SVO)      | OVP (SOV)         |
|--------------------|-------------------|
| *I ate the apple.* | *I the-apple ate.*|

When the object is a **pronoun** instead of a full noun, OVP shifts to **Verb–Subject**:

| English          | OVP        |
|------------------|------------|
| *I see him.*     | *see-I him.* (more or less) |

### 3.2 Suffixes do the heavy lifting

OVP marks grammatical roles with suffixes attached to nouns and verbs:

- A **noun used as a subject** takes a suffix indicating *proximity*:
  - `-ii` for proximal ("this / these")
  - `-uu` for distal ("that / those")
- A **noun used as an object** takes one of four suffixes depending on proximity *and* whether the noun ends in a glottal stop (`'`):
  - proximal: `-eika` (after `'`) or `-neika`
  - distal:   `-uka`  (after `'`) or `-noka`
- A **verb** takes a suffix for tense/aspect:
  - past simple `-ku`, present simple `-dü`, future simple `-wei`, present perfect `-pü`, present/past continuous `-ti`

### 3.3 Object-pronoun prefix and lenition

When a transitive verb has an object, the verb is prefixed with an *object-pronoun prefix* that agrees with the object's proximity and number, and the verb's first consonant **leniates** (softens):

`p → b`, `t → d`, `k → g`, `s → z`, `m → w̃`

So the verb stem `tüka` ("eat") with the distal-singular object prefix `u-` becomes `u-düka` (the `t` softens to `d`).

### 3.4 Pronouns

OVP distinguishes more pronouns than English, including dual ("we two"), inclusive vs. exclusive "we", and proximal vs. distal third person. We will see them as Python enum values shortly.

### 3.5 Putting it together

Here is the schema in one line:

```
SUBJECT  OBJECT-suffix  object_prefix-LENIS_VERB-tense_suffix
```

Example: *I ate that apple* &rarr; `nüü aaponu'-uka u-düka-ku`
- `nüü` &nbsp;&nbsp;&nbsp;= "I" (subject pronoun)
- `aaponu'-uka` = "apple" + distal-glottal object suffix
- `u-` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= 3rd-person distal-singular object prefix
- `düka` &nbsp;&nbsp;= leniated form of `tüka` ("eat")
- `-ku` &nbsp;&nbsp;&nbsp;= past simple

Now we can mirror all of that in Python."""
)

# ============================================================================
# Pydantic = grammar
# ============================================================================
md(
    """## 4. The big idea: a Pydantic model *is* the grammar

The package `yaduha-ovp` defines OVP grammar as a set of Python classes. The interesting move is this:

> **Every grammatically valid OVP sentence is exactly the set of values you can construct from these models. Every invalid sentence is a value the model refuses to construct.**

That is, the type system encodes the grammar. Below we import the building blocks one at a time."""
)

code(
    """from yaduha_ovp import (
    Pronoun,
    Proximity,
    Plurality,
    TenseAspect,
    SubjectNoun,
    ObjectNoun,
    TransitiveVerb,
    IntransitiveVerb,
    SubjectVerbSentence,
    SubjectVerbObjectSentence,
)

# Each grammatical category is an enum — the only allowed values are the linguistically-real ones.
print("Proximity values:", [p.value for p in Proximity])
print("Plurality values:", [p.value for p in Plurality])
print("Tense/aspect:    ", [t.value for t in TenseAspect])
print()
print("OVP pronouns:")
for p in Pronoun:
    print(f"  {p.name:25s}  ({p.value})")
"""
)

md(
    """### 4.1 A noun is a structured object

A `SubjectNoun` is not a string. It is an object with three required pieces of information that the grammar demands:

- a **head** (the lemma, like `"apple"`),
- a **proximity** (proximal or distal),
- a **plurality** (singular, dual, or plural).

You cannot construct one without choosing all three. That is the grammar enforcing itself."""
)

code(
    """coyote_distal = SubjectNoun(head="coyote", proximity=Proximity.distal, plurality=Plurality.singular)
print(coyote_distal)            # rendered with the right subject suffix
print(repr(coyote_distal))      # the underlying structured form
"""
)

md(
    """### 4.2 A verb chooses tense by construction

A `TransitiveVerb` carries a lemma *and* a tense/aspect. There is no way to forget the tense suffix &mdash; the type makes you pick one."""
)

code(
    """eat_past = TransitiveVerb(lemma="eat", tense_aspect=TenseAspect.past_simple)
see_future = TransitiveVerb(lemma="see", tense_aspect=TenseAspect.future_simple)
print(eat_past)
print(see_future)
"""
)

md(
    """### 4.3 A sentence is a structured object too

`SubjectVerbObjectSentence` requires a subject, a transitive verb, and an object. Putting them together produces a fully-formed OVP sentence &mdash; with the right suffixes, the right object prefix on the verb, and lenition applied automatically &mdash; just by calling `str(...)`."""
)

code(
    """sentence = SubjectVerbObjectSentence(
    subject=Pronoun.I,
    verb=TransitiveVerb(lemma="eat", tense_aspect=TenseAspect.past_simple),
    object=ObjectNoun(head="apple", proximity=Proximity.distal, plurality=Plurality.singular),
)

print("Built from structured pieces:")
print(sentence.model_dump_json(indent=2))
print()
print("Rendered to OVP:")
print(" ", sentence)
print()
print("English meaning: 'I ate that apple.'")
"""
)

md(
    """Notice what happened in the rendering:

- `nüü` came from the subject pronoun `I`.
- `aaponu'-uka` came from the object noun: lemma `apple` &rarr; OVP stem `aaponu'`, plus the distal-glottal object suffix `-uka`.
- `u-düka-ku` came from the verb: object-prefix `u-` (distal singular) + leniated stem `düka` (from `tüka`) + past-simple suffix `-ku`.

We did not write any of those rules in this cell. They live inside the Pydantic models, encoded once, and applied every time a sentence is rendered.

### 4.4 Try it yourself

Pick different pieces and see the sentence change. The cell below is interactive &mdash; the dropdowns rebuild a sentence on the fly. The output is always grammatical, no matter what you choose."""
)

code(
    """import ipywidgets as widgets
from IPython.display import display
from yaduha_ovp import NOUN_LOOKUP, TRANSITIVE_VERB_LOOKUP

subject_w = widgets.Dropdown(
    options=[("I (pronoun)", "I"), ("you (pronoun)", "you"), ("we two (pronoun)", "we_two"),
             ("the coyote", "coyote-distal"), ("this dog", "dog-proximal"),
             ("those mountains", "mountain-distal-plural")],
    value="I", description="Subject:",
)
verb_w = widgets.Dropdown(
    options=sorted(TRANSITIVE_VERB_LOOKUP.keys()),
    value="eat", description="Verb:",
)
tense_w = widgets.Dropdown(
    options=[(t.value, t) for t in TenseAspect],
    value=TenseAspect.past_simple, description="Tense:",
)
object_w = widgets.Dropdown(
    options=[("the apple (distal)", "apple-distal"), ("this dog (proximal)", "dog-proximal"),
             ("those mountains (distal pl)", "mountain-distal-plural"),
             ("him/her/it (distal pronoun)", "Pronoun:he_she_it_distal"),
             ("me (pronoun)", "Pronoun:I")],
    value="apple-distal", description="Object:",
)
out = widgets.Output()


def _make_subject(token):
    if token in {"I", "you", "we_two"}:
        return Pronoun[token]
    head, prox, *rest = token.split("-")
    plurality = Plurality.plural if rest and rest[0] == "plural" else Plurality.singular
    return SubjectNoun(head=head, proximity=Proximity[prox], plurality=plurality)


def _make_object(token):
    if token.startswith("Pronoun:"):
        return Pronoun[token.split(":", 1)[1]]
    head, prox, *rest = token.split("-")
    plurality = Plurality.plural if rest and rest[0] == "plural" else Plurality.singular
    return ObjectNoun(head=head, proximity=Proximity[prox], plurality=plurality)


def _render(*_):
    out.clear_output()
    sentence = SubjectVerbObjectSentence(
        subject=_make_subject(subject_w.value),
        verb=TransitiveVerb(lemma=verb_w.value, tense_aspect=tense_w.value),
        object=_make_object(object_w.value),
    )
    with out:
        print("OVP: ", sentence)
        print("Structured:")
        print(sentence.model_dump_json(indent=2))


for w in (subject_w, verb_w, tense_w, object_w):
    w.observe(_render, names="value")
_render()
display(widgets.VBox([subject_w, verb_w, tense_w, object_w, out]))
"""
)

md(
    """### 4.5 Random grammatically-valid sentences

Because the grammar is encoded in code, we can ask Python to generate as many valid sentences as we like &mdash; useful for testing, for building example sets, or just for getting a feel for the language."""
)

code(
    """for s in SubjectVerbObjectSentence.sample_iter(8):
    print(s)
"""
)

# ============================================================================
# LLM uses the grammar
# ============================================================================
md(
    """## 5. The LLM-RBMT pipeline: an LLM that *uses* the grammar

We now have two things:

1. A grammar that **always produces a valid OVP sentence** &mdash; but only if a human picks each piece (which subject? which verb? which tense?).
2. An LLM that is great at understanding English &mdash; but **invents nonsense** when asked to produce OVP directly.

The insight of **LLM-Assisted Rule-Based Machine Translation** (LLM-RBMT) is to combine them:

> Let the LLM make the *choices* (which lemma, which tense, which proximity), and let the rules render the result.

The LLM never writes OVP directly. It only writes Pydantic objects. The structured-output feature of modern LLMs guarantees that the object is well-formed &mdash; and well-formed means *grammatical*.

The full pipeline looks like this:

```
English input
    │
    ▼
[ Sentence simplifier ]   ← LLM call: "I saw him eat the apple."  →  ["He ate the apple.", "I saw him."]
    │
    ▼
[ Structured translator ]  ← LLM call with Pydantic schema → SubjectVerbObjectSentence(...)
    │
    ▼
[ Renderer ]  ← deterministic Python: turn the object into 'isha'-uu a-buni-ku' style strings
    │
    ▼
[ Back-translator ]  ← LLM call: render the structured form back into English to verify meaning
```

Steps 1 and 2 use an LLM but constrain its output. Step 3 is pure code. Step 4 lets a non-speaker user check whether their input was understood correctly &mdash; a critical feature when there are no fluent speakers around to verify.

Below we instantiate the actual `PipelineTranslator` from `yaduha`."""
)

code(
    """from yaduha.agent.openai import OpenAIAgent
from yaduha.translator.pipeline import PipelineTranslator

if OPENAI_API_KEY:
    agent = OpenAIAgent(model="gpt-4o", api_key=OPENAI_API_KEY)
    pipeline = PipelineTranslator.from_language("ovp", agent=agent)
    print("Pipeline translator ready.")
else:
    pipeline = None
    print("No API key — skipping; results from a previous run shown below.")
"""
)

md(
    """### 5.1 Run the pipeline on the same sentences that broke the naive LLM"""
)

code(
    """def show_translation(t):
    print(f"EN  : {t.source}")
    print(f"OVP : {t.target}")
    if t.back_translation:
        print(f"BACK: {t.back_translation.source}")
    print(f"      ({t.translation_time:.2f}s, {t.prompt_tokens} prompt + {t.completion_tokens} completion tokens)")
    print()


if pipeline is not None:
    for english in NAIVE_SENTENCES:
        show_translation(pipeline.translate(english))
else:
    print("(skipped — needs API key)")
"""
)

md(
    """### 5.2 Look inside the pipeline

The translator did three things internally for each input sentence:

1. Turned the English into a list of `SubjectVerbSentence` / `SubjectVerbObjectSentence` objects (the LLM, constrained by the Pydantic schema).
2. Called `str(...)` on each &mdash; pure Python, deterministic.
3. Asked an LLM to read the structured form back into English, so we have something a non-speaker can verify.

We can call the inner step directly to see the structured intermediate."""
)

code(
    """from yaduha.tool.english_to_sentences import EnglishToSentencesTool

if pipeline is not None:
    tool = EnglishToSentencesTool(agent=agent, SentenceType=(SubjectVerbSentence, SubjectVerbObjectSentence))
    response = tool("I will see the dog.")
    for sent in response.content.sentences:
        print("Structured:", sent.model_dump_json(indent=2))
        print("Rendered  :", sent)
        print()
else:
    print("(skipped — needs API key)")
"""
)

md(
    """### 5.3 Sentence simplification

What about input the grammar *cannot* directly express, like *"I saw him eat the apple"*? OVP does not have an equivalent of English's accusative-with-infinitive construction.

The pipeline handles this by asking the LLM to **simplify**: split the input into two simpler sentences that the grammar *can* express.

> *"I saw him eat the apple"* &rarr; *"He ate the apple."* + *"I saw him."*

The information loss is real but explicit, and the user can see it via the back-translation."""
)

code(
    """if pipeline is not None:
    show_translation(pipeline.translate("I saw him eat the apple."))
    show_translation(pipeline.translate("She laughed quietly at his silly joke."))
else:
    print("(skipped — needs API key)")
"""
)

# ============================================================================
# Mini experiment
# ============================================================================
md(
    """## 6. Mini experiment: Pipeline vs. Instructions

The other approach you will hear about in the papers is the **Instructions Translator**: instead of using a structured grammar tool, you stuff the entire grammar description into the LLM's system prompt and ask it to produce OVP directly. It is much easier to implement, but it is *not* guaranteed to produce grammatical output.

Let's compare them on the same inputs and see what differs."""
)

code(
    """from yaduha.translator.instructions import InstructionsTranslator

if OPENAI_API_KEY:
    instructions = InstructionsTranslator.from_language("ovp", agent=agent)
    print("Instructions translator ready.")
else:
    instructions = None
"""
)

code(
    """EXPERIMENT_SENTENCES = [
    "The dog sees the cat.",
    "I will eat the apple.",
    "We are running.",
    "The coyote chased that rabbit.",
    "You read the book.",
]

if pipeline is not None and instructions is not None:
    for english in EXPERIMENT_SENTENCES:
        print(f"=== {english} ===")
        p = pipeline.translate(english)
        i = instructions.translate(english)
        print(f"  Pipeline    : {p.target}")
        print(f"    back→EN   : {p.back_translation.source if p.back_translation else '-'}")
        print(f"  Instructions: {i.target}")
        print()
else:
    print("(skipped — needs API key)")
"""
)

md(
    """### What to look for

- **Pipeline** outputs are guaranteed to follow the schema in section 3 (subject suffix, object suffix, object-pronoun prefix, lenition, tense suffix). If you read the structured form, you can verify the grammar mechanically.
- **Instructions** outputs are often close to right but quietly drop a suffix, fail to leniate, or use a wrong object-pronoun prefix. To a non-speaker the difference is invisible.

In our published evaluation on 150 sentences across six grammatical categories, the Pipeline translator achieved the highest overall translation quality, while the Instructions translator was the easiest to implement but least reliable for production use."""
)

# ============================================================================
# Closing
# ============================================================================
md(
    """## 7. Recap and where to go next

What we did:

1. Confirmed that asking a state-of-the-art LLM to translate into OVP directly produces fluent-sounding nonsense.
2. Walked through enough OVP grammar to recognize a correct sentence.
3. Saw that the **Pydantic models in `yaduha-ovp` *are* the grammar** &mdash; constructing a valid object is exactly constructing a valid sentence.
4. Ran the **LLM-RBMT pipeline**, which lets an LLM make grammatical *choices* while the grammar enforces correctness deterministically.
5. Compared the structured pipeline against a prompt-only baseline.

### For the linguist or community member

The Yaduha framework is designed so that adding a new language means writing a new language pack &mdash; a Pydantic schema describing your grammar, a vocabulary list, and example sentences. No machine learning training required; no parallel corpus required. The grammar lives entirely in code that a linguist and a programmer can read together.

### For the programmer or researcher

- The framework: [`yaduha`](./yaduha/) &mdash; agents, translators, evaluators, language loader.
- The OVP language pack: [`yaduha-ovp`](./yaduha-ovp/) &mdash; a worked example to copy.
- The papers: [`paper_llm_rbmt/`](./paper_llm_rbmt/) (the original LLM-RBMT paper) and [`paper_llm_rbmt_2/`](./paper_llm_rbmt_2/) (the systematic evaluation across five translator strategies).

To build your own language pack, start by copying `yaduha-ovp/yaduha_ovp/` and editing `vocab.py` and the sentence classes in `__init__.py`. The `LanguageLoader.validate_language(...)` helper will check that you have all the required pieces.

> *Maanohoobüü &mdash; Thank you.*
"""
)


# ============================================================================
# Write notebook
# ============================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (uv: yaduha-tutorial)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Add an id to every cell (nbformat 5 requires it)
import secrets

for i, c in enumerate(notebook["cells"]):
    c["id"] = f"cell-{i:03d}-{secrets.token_hex(4)}"

out = Path(__file__).parent / "tutorial.ipynb"
out.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {out} ({len(cells)} cells)")
