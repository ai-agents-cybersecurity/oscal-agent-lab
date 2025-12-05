# src/oscal_agent_lab/config.py
"""Configuration for oscal-agent-lab."""

from pathlib import Path
import os

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]

# Where your cloned oscal-content repo lives
OSCAL_CONTENT_DIR = BASE_DIR / "data" / "oscal-content"

# Example catalog to index (800-53 rev5)
OSCAL_CATALOG_PATH = (
    OSCAL_CONTENT_DIR
    / "nist.gov"
    / "SP800-53"
    / "rev5"
    / "json"
    / "NIST_SP-800-53_rev5_catalog.json"
)

# Baseline profile files (contain control selections)
OSCAL_HIGH_BASELINE_PATH = (
    OSCAL_CONTENT_DIR
    / "nist.gov"
    / "SP800-53"
    / "rev5"
    / "json"
    / "NIST_SP-800-53_rev5_HIGH-baseline_profile.json"
)
OSCAL_MODERATE_BASELINE_PATH = (
    OSCAL_CONTENT_DIR
    / "nist.gov"
    / "SP800-53"
    / "rev5"
    / "json"
    / "NIST_SP-800-53_rev5_MODERATE-baseline_profile.json"
)
OSCAL_LOW_BASELINE_PATH = (
    OSCAL_CONTENT_DIR
    / "nist.gov"
    / "SP800-53"
    / "rev5"
    / "json"
    / "NIST_SP-800-53_rev5_LOW-baseline_profile.json"
)

# Example SSP paths for future DiffAgent
OSCAL_SSP_EXAMPLES_DIR = OSCAL_CONTENT_DIR / "examples" / "ssp" / "json"

# Baseline profiles (low/moderate/high)
OSCAL_PROFILES_DIR = OSCAL_CONTENT_DIR / "nist.gov" / "SP800-53" / "rev5" / "json"

# LLM / embeddings config (adjust as needed)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Vectorstore persistence (optional)
VECTORSTORE_PATH = BASE_DIR / "data" / "vectorstore"
