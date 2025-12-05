# oscal-agent-lab v0.4

| File | Purpose |
|------|---------|
| [pyproject.toml](cci:7://file:///Users/spider/Desktop/oscal-agent-lab/pyproject.toml:0:0-0:0) | Package config with all dependencies |
| [.env.example](cci:7://file:///Users/spider/Desktop/oscal-agent-lab/.env.example:0:0-0:0) | Template for API keys |
| [src/oscal_agent_lab/config.py](cci:7://file:///Users/spider/Desktop/oscal-agent-lab/src/oscal_agent_lab/config.py:0:0-0:0) | Model + path configuration |
| [src/oscal_agent_lab/oscal_loader.py](cci:7://file:///Users/spider/Desktop/oscal-agent-lab/src/oscal_agent_lab/oscal_loader.py:0:0-0:0) | Parses OSCAL JSON -> LangChain Documents |
| [src/oscal_agent_lab/vectorstore.py](cci:7://file:///Users/spider/Desktop/oscal-agent-lab/src/oscal_agent_lab/vectorstore.py:0:0-0:0) | FAISS index with caching |
| [src/oscal_agent_lab/agents/explainer.py](cci:7://file:///Users/spider/Desktop/oscal-agent-lab/src/oscal_agent_lab/agents/explainer.py:0:0-0:0) | RAG-based Q&A node |
| [src/oscal_agent_lab/graph.py](cci:7://file:///Users/spider/Desktop/oscal-agent-lab/src/oscal_agent_lab/graph.py:0:0-0:0) | LangGraph workflow |
| [src/oscal_agent_lab/cli.py](cci:7://file:///Users/spider/Desktop/oscal-agent-lab/src/oscal_agent_lab/cli.py:0:0-0:0) | Interactive REPL |
| `data/oscal-content/` | NIST OSCAL content (submodule) |

## Quick verification

- **1,196 control documents** parsed from NIST SP 800-53 Rev 5
- Control families: AC, AT, AU, CA, CM, CP, IA, IR, MA, MP, PE, PL, PM, PS, PT, RA, SA, SC, SI, SR

## To run it

```bash
# Install dependencies
pip install -e .

# Set your API key
export OPENAI_API_KEY=sk-...

# Launch the CLI
python -m oscal_agent_lab.cli
```

Then ask: *"What does AC-2 require?"* or *"Which controls cover audit logging?"*

## Compare SSPs (v0.2)

```bash
# In the CLI:
diff data/test-ssps/ssp_v1.json data/test-ssps/ssp_v2.json

# Or standalone:
PYTHONPATH=src python -m oscal_agent_lab.diff_cli data/test-ssps/ssp_v1.json data/test-ssps/ssp_v2.json
```

## Build Profiles (v0.3)

```bash
# In the CLI:
profile A healthcare SaaS application handling PHI data

# With specific baseline:
profile --baseline high A federal agency financial system
```

Generates valid OSCAL profile JSON and saves to `generated_profile.json`.

## Validate OSCAL (v0.4)

```bash
# In the CLI:
validate data/oscal-content/examples/ssp/json/ssp-example.json

# Validates against official NIST OSCAL schemas
# Auto-detects model type (catalog, profile, ssp, etc.)
# Explains errors with AI-powered suggestions
```

## What's implemented

| Agent | Status | Description |
|-------|--------|-------------|
| ExplainerAgent | [+] v0.1 | RAG Q&A over 800-53 controls |
| DiffAgent | [+] v0.2 | Compare SSPs, summarize changes |
| ProfileBuilderAgent | [+] v0.3 | NL -> OSCAL profile generation |
| ValidatorAgent | [+] v0.4 | Schema validation + error explanation |
