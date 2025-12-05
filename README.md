# oscal-agent-lab

Experimental lab to explore **NIST OSCAL + AI agents** using **LangGraph** and **LangChain**.

> One repo to play with: OSCAL catalogs, profiles, SSPs, digital-twin-ish ideas, and agentic AI.

## What it does (v0.4)

- Loads the official NIST SP 800-53 Rev 5 OSCAL catalog (JSON) from the
  [`usnistgov/oscal-content`](https://github.com/usnistgov/oscal-content) repo.
- Flattens controls into text chunks and builds a vector index (1196 controls).
- Exposes four LangGraph-based agents:
  - **ExplainerAgent** - RAG-based Q&A over 800-53 controls
  - **DiffAgent** - Compare two OSCAL SSPs and summarize changes
  - **ProfileBuilderAgent** - Generate OSCAL profiles from natural language descriptions
  - **ValidatorAgent** - Schema validation with LLM-powered error explanations

> Note: This is a sandbox - expect rough edges, but also a lot of room to experiment

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

## Run the CLI

```bash
python -m oscal_agent_lab.cli
```

### Commands

**Q&A (ExplainerAgent)** - just type your question:
- `Explain AC-2 in plain language`
- `Which controls are about audit logging?`
- `What are the main responsibilities behind the IR-4 family?`

**Diff (DiffAgent)** - compare two SSPs:
- `diff path/to/ssp1.json path/to/ssp2.json`

**Profile (ProfileBuilderAgent)** - generate OSCAL profiles:
- `profile healthcare compliance requirements`
- `profile --baseline moderate cloud security controls`
- Baselines: `low` (149 controls), `moderate` (287), `high` (370)

**Validate (ValidatorAgent)** - validate OSCAL files:
- `validate path/to/file.json`

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
        diff.py           # DiffAgent (v0.2)
        profile_builder.py# ProfileBuilderAgent (v0.3)
        validator.py      # ValidatorAgent (v0.4)
      graph.py            # LangGraph graph wiring the agents
      cli.py              # simple CLI "copilot" entry point
```

## Roadmap

- [x] **ExplainerAgent** (v0.1) - Q&A over 800-53 controls using RAG
- [x] **DiffAgent** (v0.2) - compare two OSCAL SSPs and summarize what changed
- [x] **ProfileBuilderAgent** (v0.3) - generate OSCAL profiles from natural language
- [x] **ValidatorAgent** (v0.4) - schema validation with LLM error explanations
- [ ] **Digital twin experiments** - sync live state with OSCAL SSP and detect drift

## Resources

- [NIST OSCAL](https://pages.nist.gov/OSCAL/) - official OSCAL documentation
- [usnistgov/oscal-content](https://github.com/usnistgov/oscal-content) - NIST OSCAL examples
- [LangGraph docs](https://langchain-ai.github.io/langgraph/) - agent orchestration
- [OSCAL JSON Reference](https://pages.nist.gov/OSCAL-Reference/) - schema reference

---

Again, as mentioned above this is a sandbox - expect rough edges, but also a lot of room to experiment.
