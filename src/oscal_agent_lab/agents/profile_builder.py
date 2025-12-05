# src/oscal_agent_lab/agents/profile_builder.py
"""ProfileBuilderAgent: Generate OSCAL profiles from natural language."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from ..config import (
    OPENAI_MODEL,
    OSCAL_CATALOG_PATH,
    OSCAL_LOW_BASELINE_PATH,
    OSCAL_MODERATE_BASELINE_PATH,
    OSCAL_HIGH_BASELINE_PATH,
)
from ..oscal_loader import load_oscal_json
from ..vectorstore import get_retriever


# Control family descriptions for context
CONTROL_FAMILIES = {
    "ac": "Access Control - user access, authentication, separation of duties",
    "at": "Awareness and Training - security training, role-based training",
    "au": "Audit and Accountability - logging, audit records, monitoring",
    "ca": "Assessment, Authorization, Monitoring - security assessments, POA&M",
    "cm": "Configuration Management - baseline configs, change control",
    "cp": "Contingency Planning - backup, recovery, alternate sites",
    "ia": "Identification and Authentication - user IDs, MFA, credentials",
    "ir": "Incident Response - incident handling, reporting, testing",
    "ma": "Maintenance - system maintenance, remote maintenance",
    "mp": "Media Protection - media access, storage, transport, sanitization",
    "pe": "Physical and Environmental Protection - physical access, monitoring",
    "pl": "Planning - security plans, rules of behavior",
    "pm": "Program Management - risk management strategy, enterprise architecture",
    "ps": "Personnel Security - screening, termination, transfers",
    "pt": "PII Processing and Transparency - privacy notices, consent",
    "ra": "Risk Assessment - risk assessments, vulnerability scanning",
    "sa": "System and Services Acquisition - SDLC, supply chain, testing",
    "sc": "System and Communications Protection - encryption, boundary protection",
    "si": "System and Information Integrity - malware, patching, monitoring",
    "sr": "Supply Chain Risk Management - supply chain controls",
}


def get_all_control_ids() -> list[str]:
    """Get all control IDs from the catalog."""
    catalog_json = load_oscal_json(OSCAL_CATALOG_PATH)
    catalog = catalog_json.get("catalog", catalog_json)
    control_ids = []

    def extract_ids(controls: list[dict], prefix: str = ""):
        for ctrl in controls:
            ctrl_id = ctrl.get("id", "")
            if ctrl_id:
                control_ids.append(ctrl_id)
            # Handle enhancements (nested controls)
            nested = ctrl.get("controls", [])
            if nested:
                extract_ids(nested)

    for group in catalog.get("groups", []):
        extract_ids(group.get("controls", []))

    return sorted(control_ids)


def get_baseline_controls(baseline: str) -> list[str]:
    """Get control IDs from a standard baseline."""
    baseline_paths = {
        "low": OSCAL_LOW_BASELINE_PATH,
        "moderate": OSCAL_MODERATE_BASELINE_PATH,
        "high": OSCAL_HIGH_BASELINE_PATH,
    }

    path = baseline_paths.get(baseline.lower())
    if not path or not path.exists():
        return []

    profile_json = load_oscal_json(path)
    profile = profile_json.get("profile", profile_json)
    control_ids = []

    for imp in profile.get("imports", []):
        for inc in imp.get("include-controls", []):
            control_ids.extend(inc.get("with-ids", []))

    return sorted(control_ids)


# LLM setup
_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    return _llm


_PROFILE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert OSCAL security consultant helping build security profiles.

Given a system description, you select appropriate NIST SP 800-53 controls.

Control families available:
{families}

Guidelines:
1. Start with the specified baseline (low/moderate/high) as a foundation
2. ADD controls needed for the specific system context
3. Consider industry regulations (HIPAA, PCI-DSS, FedRAMP, etc.)
4. Consider data sensitivity and system criticality
5. Return ONLY a JSON object with your selections

Output format (strict JSON):
{{
  "baseline": "moderate",
  "additional_controls": ["control-id-1", "control-id-2"],
  "removed_controls": ["control-id-to-remove"],
  "rationale": "Brief explanation of key decisions"
}}""",
        ),
        (
            "human",
            """System Description:
{description}

Starting Baseline: {baseline}

Relevant control context from catalog:
{context}

Available control IDs in the {baseline} baseline:
{baseline_sample}

Select controls for this system. Return ONLY valid JSON.""",
        ),
    ]
)


def analyze_system_for_controls(
    description: str,
    baseline: str = "moderate",
) -> dict[str, Any]:
    """
    Use LLM to analyze system description and recommend controls.

    Returns dict with baseline, additional_controls, removed_controls, rationale.
    """
    llm = _get_llm()
    retriever = get_retriever(k=10)

    # Get relevant control snippets
    docs = retriever.invoke(description)
    context = "\n\n".join(d.page_content[:500] for d in docs)

    # Get baseline controls for reference
    baseline_controls = get_baseline_controls(baseline)
    baseline_sample = ", ".join(baseline_controls[:50]) + "..." if len(baseline_controls) > 50 else ", ".join(baseline_controls)

    # Format families
    families_text = "\n".join(f"- {k.upper()}: {v}" for k, v in CONTROL_FAMILIES.items())

    prompt_messages = _PROFILE_PROMPT.format_messages(
        families=families_text,
        description=description,
        baseline=baseline,
        context=context,
        baseline_sample=baseline_sample,
    )

    result = llm.invoke(prompt_messages)

    # Parse JSON from response
    try:
        # Try to extract JSON from the response
        content = result.content
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content.strip())
    except json.JSONDecodeError:
        # Return a default structure if parsing fails
        return {
            "baseline": baseline,
            "additional_controls": [],
            "removed_controls": [],
            "rationale": f"Analysis complete. Raw response: {result.content[:200]}",
        }


def generate_oscal_profile(
    title: str,
    control_ids: list[str],
    catalog_href: str = "#catalog",
) -> dict[str, Any]:
    """
    Generate a valid OSCAL profile JSON structure.

    Args:
        title: Profile title
        control_ids: List of control IDs to include
        catalog_href: Reference to the source catalog

    Returns:
        Valid OSCAL profile dict
    """
    now = datetime.now(timezone.utc).isoformat()

    profile = {
        "profile": {
            "uuid": str(uuid.uuid4()),
            "metadata": {
                "title": title,
                "last-modified": now,
                "version": "1.0.0",
                "oscal-version": "1.1.1",
                "remarks": "Generated by oscal-agent-lab ProfileBuilderAgent",
            },
            "imports": [
                {
                    "href": catalog_href,
                    "include-controls": [
                        {
                            "with-ids": sorted(control_ids),
                        }
                    ],
                }
            ],
        }
    }

    return profile


def build_profile(
    system_description: str,
    baseline: str = "moderate",
    profile_title: str | None = None,
) -> dict[str, Any]:
    """
    Generate an OSCAL profile from a system description.

    Args:
        system_description: Natural language description of the system.
        baseline: Starting baseline (low, moderate, high).
        profile_title: Optional title for the profile.

    Returns:
        Dict containing:
        - profile: Valid OSCAL profile JSON
        - analysis: LLM analysis with rationale
        - control_count: Number of controls selected
    """
    # Get baseline controls
    baseline_controls = set(get_baseline_controls(baseline))

    # Analyze system for additional/removed controls
    analysis = analyze_system_for_controls(system_description, baseline)

    # Build final control set
    final_controls = baseline_controls.copy()

    # Add recommended controls
    for ctrl in analysis.get("additional_controls", []):
        final_controls.add(ctrl)

    # Remove controls if specified
    for ctrl in analysis.get("removed_controls", []):
        final_controls.discard(ctrl)

    # Generate title if not provided
    if not profile_title:
        profile_title = f"Custom Profile - {baseline.capitalize()} Baseline Modified"

    # Generate OSCAL profile
    profile = generate_oscal_profile(
        title=profile_title,
        control_ids=list(final_controls),
        catalog_href="https://github.com/usnistgov/oscal-content/raw/main/nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json",
    )

    return {
        "profile": profile,
        "analysis": analysis,
        "control_count": len(final_controls),
        "baseline_count": len(baseline_controls),
        "added_count": len(analysis.get("additional_controls", [])),
        "removed_count": len(analysis.get("removed_controls", [])),
    }


def profile_builder_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: propose an OSCAL profile based on system description.

    Expects state with:
    - system_description: Natural language description
    - baseline: Optional baseline (default: moderate)

    Or parses from the last user message if it looks like a profile request.
    """
    messages = state.get("messages", [])

    # Check if description is directly in state
    system_description = state.get("system_description")
    baseline = state.get("baseline", "moderate")

    if not system_description:
        # Try to extract from last message
        if messages:
            last = messages[-1]
            if isinstance(last, HumanMessage):
                content = last.content.lower()
                # Check if this looks like a profile request
                if any(kw in content for kw in ["profile", "baseline", "build", "create", "generate"]):
                    system_description = last.content
                else:
                    return {
                        "messages": [
                            AIMessage(
                                content="To build a profile, describe your system. For example:\n"
                                "'Build a profile for a healthcare SaaS application handling PHI'\n"
                                "or use `build_profile()` directly."
                            )
                        ]
                    }
        else:
            return {}

    try:
        result = build_profile(system_description, baseline)

        response = f"""## OSCAL Profile Generated

**Based on:** {baseline.capitalize()} baseline
**Total controls:** {result['control_count']} (baseline: {result['baseline_count']})
**Added:** {result['added_count']} | **Removed:** {result['removed_count']}

### Analysis
{result['analysis'].get('rationale', 'No rationale provided.')}

### Additional Controls Selected
{', '.join(result['analysis'].get('additional_controls', [])) or 'None'}

### Controls Removed
{', '.join(result['analysis'].get('removed_controls', [])) or 'None'}

---
*Profile JSON generated. Use `build_profile()` directly to get the full OSCAL JSON output.*
"""

        return {
            "messages": [AIMessage(content=response)],
            "profile_result": result,
        }

    except Exception as e:
        return {"messages": [AIMessage(content=f"Error building profile: {e}")]}
