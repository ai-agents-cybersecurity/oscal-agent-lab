Let's turn **`oscal-agent-lab`** into a real thing. [test]

Below is a concrete plan + repo scaffold + starter code so you can literally copy-paste into a GitHub repo and start hacking.

---

## 1. Project concept: `oscal-agent-lab`

**One-liner:**

> A LangGraph-powered multi-agent "copilot" for OSCAL catalogs, profiles, and SSPs, using real NIST OSCAL JSON as the knowledge base.

Core ideas:

* **ExplainerAgent** - Q&A + explanations over:

  * 800-53 catalogs (controls, statements)
  * profiles / baselines
  * system security plans (SSPs)
    using NIST's official OSCAL content repo as source. ([GitHub][1])
* **DiffAgent** - compare two SSPs and summarize what changed (controls added/removed/changed).
* **ProfileBuilderAgent** - from a natural-language description of a system, propose a draft OSCAL profile (subset of controls).
* **ValidatorAgent** - validate generated OSCAL against the official JSON schema and give human-readable errors. ([NIST Pages][2])

You can build it in stages. v0.1 = ExplainerAgent, then layer the rest.

---

## 2. Repo scaffold

Suggested structure:

```text
oscal-agent-lab/
  README.md
  pyproject.toml          # or requirements.txt if you prefer
  .env.example            # API keys etc.
  data/
    oscal-content/        # git submodule or copy of usnistgov/oscal-content
      ...                 # e.g. nist.gov/SP800-53/rev5/json/...
  src/
    oscal_agent_lab/
      __init__.py
      config.py           # model + paths config
      oscal_loader.py     # load + flatten OSCAL JSON into documents
      vectorstore.py      # build / load embeddings index
      agents/
        __init__.py
        explainer.py      # ExplainerAgent
        diff.py           # DiffAgent (later)
        profile_builder.py
        validator.py
      graph.py            # LangGraph graph wiring the agents
      cli.py              # simple CLI "copilot" entry point
```

### Dependencies (minimal)

```bash
pip install -U langgraph langchain langchain-community langchain-openai
# plus your embedding model deps:
pip install -U faiss-cpu      # if you use FAISS
```

* `langgraph` for the agent workflow orchestration ([LangChain Docs][3])
* `langchain` / `langchain-openai` (or your provider of choice) for LLM & tools ([LangChain Docs][4])
* `langchain-community` for vectorstore integrations like FAISS. ([PyPI][5])

### OSCAL data source

Add NIST's OSCAL examples as a submodule or just clone:

```bash
git submodule add https://github.com/usnistgov/oscal-content.git data/oscal-content
```

This repo contains OSCAL **catalogs, profiles, SSPs, etc.** in XML/JSON/YAML, including 800-53 Rev 5 catalogs and baselines. ([GitHub][1])

Example JSON paths you can point to:

* `data/oscal-content/nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json` (catalog) ([GitHub][1])
* `data/oscal-content/examples/ssp/json/` for sample SSPs. ([NIST Pages][6])

---

## 3. v0.1: ExplainerAgent (OSCAL Copilot)

Let's get you to a **working** first version:

### 3.1 `config.py`

```python
# src/oscal_agent_lab/config.py
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[2]

# Where your cloned oscal-content repo lives
OSCAL_CONTENT_DIR = BASE_DIR / "data" / "oscal-content"

# Example catalog to index (800-53 rev5)
OSCAL_CATALOG_PATH = (
    OSCAL_CONTENT_DIR / "nist.gov" / "SP800-53" / "rev5" / "json" /
    "NIST_SP-800-53_rev5_catalog.json"
)

# LLM / embeddings config (adjust as needed)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
```

### 3.2 `oscal_loader.py` - load + flatten OSCAL catalog

We'll convert an OSCAL catalog (e.g. 800-53) into `Document` chunks, roughly one per control.

```python
# src/oscal_agent_lab/oscal_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document  # v1-style docs API

def load_oscal_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def catalog_to_documents(catalog_json: Dict[str, Any]) -> List[Document]:
    """
    Flatten an OSCAL catalog into Documents (roughly one per control).

    This assumes the typical OSCAL catalog layout:
      { "catalog": { "groups": [ { "id": ..., "controls": [ ... ] }, ... ] } }
    as used in the NIST SP 800-53 Rev 5 OSCAL examples. :contentReference[oaicite:8]{index=8}
    """
    catalog = catalog_json.get("catalog", catalog_json)
    docs: List[Document] = []

    groups = catalog.get("groups", [])
    for group in groups:
        group_id = group.get("id")
        for control in group.get("controls", []):
            control_id = control.get("id")
            title = control.get("title", "")
            parts = control.get("parts", [])

            text_parts: List[str] = []
            if title:
                text_parts.append(f"Control {control_id}: {title}")

            # Basic prose extraction from parts
            for part in parts:
                name = part.get("name") or part.get("title")
                prose = part.get("prose")
                if prose:
                    if name:
                        text_parts.append(f"{name}: {prose}")
                    else:
                        text_parts.append(prose)

            if not text_parts:
                continue

            content = "\n\n".join(text_parts)
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "control_id": control_id,
                        "group_id": group_id,
                        "source": "oscal_catalog",
                    },
                )
            )

    return docs
```

### 3.3 `vectorstore.py` - build a simple FAISS index

```python
# src/oscal_agent_lab/vectorstore.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_community.vectorstores.faiss import FAISS  # community integration :contentReference[oaicite:9]{index=9}
from langchain_openai import OpenAIEmbeddings

from .config import OSCAL_CATALOG_PATH, OPENAI_EMBED_MODEL
from .oscal_loader import load_oscal_json, catalog_to_documents

_vectorstore: Optional[FAISS] = None

def get_vectorstore() -> FAISS:
    """
    Lazy-build and cache a FAISS vectorstore from the OSCAL catalog.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    catalog_json = load_oscal_json(OSCAL_CATALOG_PATH)
    docs = catalog_to_documents(catalog_json)

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    _vectorstore = FAISS.from_documents(docs, embeddings)
    return _vectorstore

def get_retriever(k: int = 5):
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})
```

> You can later generalize this to also ingest **profiles** and **SSPs** by adding similar flatteners for those OSCAL models. ([NIST Pages][2])

### 3.4 `agents/explainer.py` - the ExplainerAgent node

We'll build a node that:

1. Takes the latest user question from state.
2. Retrieves relevant OSCAL control docs.
3. Calls the LLM with those docs as context.

```python
# src/oscal_agent_lab/agents/explainer.py
from __future__ import annotations

from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage, HumanMessage

from ..config import OPENAI_MODEL
from ..vectorstore import get_retriever

# Init model + retriever once
_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
_retriever = get_retriever(k=6)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an OSCAL cybersecurity and compliance copilot. "
                "You answer questions about NIST SP 800-53 and related OSCAL content. "
                "Use only the provided OSCAL snippets as ground truth. "
                "When helpful, refer to controls by their IDs (e.g., AC-2)."
            ),
        ),
        (
            "human",
            "User question:\n{question}\n\n"
            "Relevant OSCAL snippets:\n{context}\n\n"
            "Answer in clear, concise language."
        ),
    ]
)

def explainer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: uses RAG over OSCAL catalog to answer the latest user question.
    Expects a MessagesState-like dict with `messages` list.
    """
    messages: list[AnyMessage] = state["messages"]
    if not messages:
        return {}

    last = messages[-1]
    if not isinstance(last, HumanMessage):
        # Only respond when the last message is from the user
        return {}

    question = last.content
    docs = _retriever.get_relevant_documents(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = _PROMPT.format(question=question, context=context)
    result = _llm.invoke(prompt.to_messages())

    # LangGraph will merge this into state (we're using add_messages reducer)
    return {"messages": [result]}
```

### 3.5 `graph.py` - wiring ExplainerAgent in LangGraph

We'll use LangGraph's `StateGraph` and built-in `MessagesState` for chat. ([LangChain Docs][7])

```python
# src/oscal_agent_lab/graph.py
from __future__ import annotations

from langgraph.graph import StateGraph, MessagesState, START, END

from .agents.explainer import explainer_node

def build_graph():
    """
    Build the simplest graph: START -> explainer -> END
    using LangGraph's MessagesState.
    """
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("explainer", explainer_node)
    graph_builder.add_edge(START, "explainer")
    graph_builder.add_edge("explainer", END)

    return graph_builder.compile()
```

### 3.6 `cli.py` - little REPL to play with it

```python
# src/oscal_agent_lab/cli.py
from __future__ import annotations

from langchain_core.messages import HumanMessage, AIMessage
from .graph import build_graph

def main():
    graph = build_graph()
    state = {"messages": []}

    print("OSCAL Agent Lab - Explainer v0.1")
    print("Type your question (or 'exit'):")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = graph.invoke(state)

        # Get last AI message
        ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            print("\nAgent:", ai_messages[-1].content, "\n")

if __name__ == "__main__":
    main()
```

Run it with:

```bash
export OPENAI_API_KEY=...   # or your equivalent provider env
python -m oscal_agent_lab.cli
```

You now have:

* an **OSCAL-aware RAG copilot** over 800-53 Rev 5 controls,
* implemented as a **LangGraph agent node**,
* with clear extension points for more agents.

---

## 4. Next steps (to keep it "cool")

Once v0.1 works, here's how to level it up:

### 4.1 Add **DiffAgent** (SSP comparison)

* Loader: flatten SSP JSON into per-control summaries (implemented requirements, by-component). ([NIST Pages][2])
* Node:

  * Input: file A path, file B path, or two loaded objects.
  * Output:

    * controls added / removed / changed
    * a short risk narrative ("this change weakens X", "this improves Y") via LLM.

Graph: add a branching path based on user intent ("compare these two SSPs").

### 4.2 Add **ProfileBuilderAgent**

* Input: a natural language description of the system, sector, data sensitivity.
* Node:

  * Use catalog + maybe existing baseline profiles (e.g. low/moderate/high from `oscal-content`) to propose a subset. ([GitHub][8])
  * Output: valid OSCAL **profile JSON** skeleton.

### 4.3 Add **ValidatorAgent**

* Use official **OSCAL JSON schema** / metaschema reference to validate generated catalogs/profiles/SSPs and return structured errors. ([NIST Pages][2])
* Optionally loop back into the LLM: "fix this OSCAL according to these schema errors".

---

## 5. README starter (you can paste into GitHub)

Here's a minimal `README.md` skeleton:

````markdown
# oscal-agent-lab

Experimental lab to explore **NIST OSCAL + AI agents** using **LangGraph** and **LangChain**.

> One repo to play with: OSCAL catalogs, profiles, SSPs, digital-twin-ish ideas, and agentic AI.

## What it does (v0.1)

- Loads the official NIST SP 800-53 Rev 5 OSCAL catalog (JSON) from the
  [`usnistgov/oscal-content`](https://github.com/usnistgov/oscal-content) repo.
- Flattens controls into text chunks and builds a vector index.
- Exposes a LangGraph-based **ExplainerAgent**:
  - Ask natural-language questions about controls (e.g. "What does AC-2 actually require?").
  - Get answers grounded in the official OSCAL content.

Under the hood it uses:

- **LangGraph** for agent workflow orchestration  
- **LangChain** for LLM + retrieval plumbing  
- **FAISS** (via `langchain-community`) for local semantic search over OSCAL docs.

## Install

```bash
git clone https://github.com/<you>/oscal-agent-lab.git
cd oscal-agent-lab

# OSCAL examples (NIST)
git submodule add https://github.com/usnistgov/oscal-content.git data/oscal-content

# Python deps
pip install -U langgraph langchain langchain-community langchain-openai faiss-cpu
````

Set your model credentials (example with OpenAI):

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_EMBED_MODEL=text-embedding-3-small
```

## Run the OSCAL Explainer

```bash
python -m oscal_agent_lab.cli
```

Then ask things like:

* `Explain AC-2 in plain language`
* `Which controls are about audit logging?`
* `What are the main responsibilities behind the IR-4 family?`

## Roadmap

* **DiffAgent** - compare two OSCAL SSPs and summarize what changed.
* **ProfileBuilderAgent** - propose OSCAL profiles from natural language ("I'm a SaaS handling healthcare data...").
* **ValidatorAgent** - run schema validation and explain OSCAL errors in human terms.
* **Digital twin experiments** - sync a small "live state" with an OSCAL SSP and use agents to spot drift.

---

This is a sandbox - expect rough edges, but also a lot of room to experiment.

```

---

If you'd like, I can next:

- sketch the **DiffAgent** node logic, or  
- design a simple **Streamlit/FastAPI** UI so people can upload their own OSCAL files and chat with the agent.
::contentReference[oaicite:15]{index=15}
```

[1]: https://github.com/usnistgov/oscal-content?utm_source=chatgpt.com "NIST SP 800-53 content and other OSCAL content examples"
[2]: https://pages.nist.gov/OSCAL-Reference/models/v1.1.1/system-security-plan/json-reference/?utm_source=chatgpt.com "System Security Plan Model v1.1.1 JSON Format Reference"
[3]: https://docs.langchain.com/oss/python/langgraph/install?utm_source=chatgpt.com "Install LangGraph - Docs by LangChain"
[4]: https://docs.langchain.com/oss/python/langchain/install?utm_source=chatgpt.com "Install LangChain"
[5]: https://pypi.org/project/langchain-community/?utm_source=chatgpt.com "langchain-community"
[6]: https://pages.nist.gov/OSCAL/learn/concepts/layer/implementation/ssp/?utm_source=chatgpt.com "System Security Plan Model (SSP) - NIST Pages"
[7]: https://docs.langchain.com/oss/python/langgraph/overview?utm_source=chatgpt.com "LangGraph overview - Docs by LangChain"
[8]: https://github.com/usnistgov/oscal-content/blob/master/nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_PRIVACY-baseline_profile.json?utm_source=chatgpt.com "NIST_SP-800-53_rev5_PRIVACY-baseline_profile.json"
