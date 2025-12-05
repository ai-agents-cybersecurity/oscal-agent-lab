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
pip install -e .
# or without editable install:
pip install -U langgraph langchain langchain-community langchain-openai faiss-cpu
```

Set your model credentials (example with OpenAI):

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_EMBED_MODEL=text-embedding-3-small
```

Or copy `.env.example` to `.env` and fill in your values.

## Run the OSCAL Explainer

```bash
python -m oscal_agent_lab.cli
```

Then ask things like:

- `Explain AC-2 in plain language`
- `Which controls are about audit logging?`
- `What are the main responsibilities behind the IR-4 family?`

## Project Structure

```text
oscal-agent-lab/
  README.md
  pyproject.toml
  .env.example
  data/
    oscal-content/        # git submodule: usnistgov/oscal-content
  src/
    oscal_agent_lab/
      __init__.py
      config.py           # model + paths config
      oscal_loader.py     # load + flatten OSCAL JSON into documents
      vectorstore.py      # build / load embeddings index
      agents/
        __init__.py
        explainer.py      # ExplainerAgent (v0.1)
        diff.py           # DiffAgent (planned)
        profile_builder.py# ProfileBuilderAgent (planned)
        validator.py      # ValidatorAgent (planned)
      graph.py            # LangGraph graph wiring the agents
      cli.py              # simple CLI "copilot" entry point
```

## Roadmap

- [x] **ExplainerAgent** - Q&A over 800-53 controls using RAG
- [ ] **DiffAgent** - compare two OSCAL SSPs and summarize what changed
- [ ] **ProfileBuilderAgent** - propose OSCAL profiles from natural language
- [ ] **ValidatorAgent** - run schema validation and explain OSCAL errors
- [ ] **Digital twin experiments** - sync live state with OSCAL SSP and detect drift

## Resources

- [NIST OSCAL](https://pages.nist.gov/OSCAL/) - official OSCAL documentation
- [usnistgov/oscal-content](https://github.com/usnistgov/oscal-content) - NIST OSCAL examples
- [LangGraph docs](https://langchain-ai.github.io/langgraph/) - agent orchestration
- [OSCAL JSON Reference](https://pages.nist.gov/OSCAL-Reference/) - schema reference

---

This is a sandbox - expect rough edges, but also a lot of room to experiment. [test]
