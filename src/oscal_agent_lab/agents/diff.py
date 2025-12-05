# src/oscal_agent_lab/agents/diff.py
"""DiffAgent: Compare two OSCAL SSPs and summarize changes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from ..config import OPENAI_MODEL
from ..oscal_loader import load_oscal_json


@dataclass
class ControlImplementation:
    """Represents an implemented requirement from an SSP."""

    control_id: str
    statements: list[dict[str, Any]] = field(default_factory=list)
    by_components: list[dict[str, Any]] = field(default_factory=list)
    props: list[dict[str, Any]] = field(default_factory=list)
    remarks: str = ""

    def get_description_text(self) -> str:
        """Extract all description text from this implementation."""
        texts = []

        # From statements
        for stmt in self.statements:
            stmt_id = stmt.get("statement-id", "")
            for comp in stmt.get("by-components", []):
                desc = comp.get("description", "")
                if desc:
                    texts.append(f"[{stmt_id}] {desc}")

        # From by-components at control level
        for comp in self.by_components:
            desc = comp.get("description", "")
            if desc:
                texts.append(desc)

        if self.remarks:
            texts.append(f"Remarks: {self.remarks}")

        return "\n".join(texts) if texts else "(no description)"


@dataclass
class SSPDiff:
    """Result of comparing two SSPs."""

    ssp_a_name: str
    ssp_b_name: str
    added_controls: list[str] = field(default_factory=list)
    removed_controls: list[str] = field(default_factory=list)
    changed_controls: list[tuple[str, str, str]] = field(default_factory=list)  # (id, old, new)
    unchanged_controls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ssp_a": self.ssp_a_name,
            "ssp_b": self.ssp_b_name,
            "added_controls": self.added_controls,
            "removed_controls": self.removed_controls,
            "changed_controls": [
                {"control_id": c[0], "before": c[1], "after": c[2]}
                for c in self.changed_controls
            ],
            "unchanged_controls": self.unchanged_controls,
            "stats": {
                "added": len(self.added_controls),
                "removed": len(self.removed_controls),
                "changed": len(self.changed_controls),
                "unchanged": len(self.unchanged_controls),
            },
        }


def extract_implementations(ssp_json: dict[str, Any]) -> dict[str, ControlImplementation]:
    """Extract all control implementations from an SSP."""
    ssp = ssp_json.get("system-security-plan", ssp_json)
    ctrl_impl = ssp.get("control-implementation", {})
    impl_reqs = ctrl_impl.get("implemented-requirements", [])

    implementations: dict[str, ControlImplementation] = {}

    for req in impl_reqs:
        control_id = req.get("control-id", "")
        if not control_id:
            continue

        impl = ControlImplementation(
            control_id=control_id,
            statements=req.get("statements", []),
            by_components=req.get("by-components", []),
            props=req.get("props", []),
            remarks=req.get("remarks", ""),
        )
        implementations[control_id] = impl

    return implementations


def get_ssp_metadata(ssp_json: dict[str, Any]) -> dict[str, str]:
    """Extract metadata from an SSP."""
    ssp = ssp_json.get("system-security-plan", ssp_json)
    metadata = ssp.get("metadata", {})
    sys_chars = ssp.get("system-characteristics", {})

    return {
        "title": metadata.get("title", "Unknown"),
        "version": metadata.get("version", "unknown"),
        "system_name": sys_chars.get("system-name", metadata.get("title", "Unknown System")),
    }


def compare_ssps(
    ssp_a_path: str | Path,
    ssp_b_path: str | Path,
) -> SSPDiff:
    """
    Compare two OSCAL SSPs and identify differences.

    Args:
        ssp_a_path: Path to the "before" SSP
        ssp_b_path: Path to the "after" SSP

    Returns:
        SSPDiff object with added, removed, changed, and unchanged controls.
    """
    ssp_a_json = load_oscal_json(ssp_a_path)
    ssp_b_json = load_oscal_json(ssp_b_path)

    meta_a = get_ssp_metadata(ssp_a_json)
    meta_b = get_ssp_metadata(ssp_b_json)

    impl_a = extract_implementations(ssp_a_json)
    impl_b = extract_implementations(ssp_b_json)

    controls_a = set(impl_a.keys())
    controls_b = set(impl_b.keys())

    diff = SSPDiff(
        ssp_a_name=meta_a["system_name"],
        ssp_b_name=meta_b["system_name"],
    )

    # Added in B (not in A)
    diff.added_controls = sorted(controls_b - controls_a)

    # Removed from A (not in B)
    diff.removed_controls = sorted(controls_a - controls_b)

    # Check for changes in common controls
    common = controls_a & controls_b
    for ctrl_id in sorted(common):
        text_a = impl_a[ctrl_id].get_description_text()
        text_b = impl_b[ctrl_id].get_description_text()

        if text_a != text_b:
            diff.changed_controls.append((ctrl_id, text_a, text_b))
        else:
            diff.unchanged_controls.append(ctrl_id)

    return diff


# LLM for generating summaries
_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    return _llm


_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an OSCAL security analyst. Your job is to analyze differences between 
two System Security Plans (SSPs) and provide a clear, actionable summary.

Focus on:
- Security implications of added/removed controls
- Risk changes from modified implementations
- Compliance impact
- Recommendations for review

Be concise but thorough. Use control IDs when referencing specific controls.""",
        ),
        (
            "human",
            """Compare these two SSPs and summarize the changes:

**SSP A (Before):** {ssp_a_name}
**SSP B (After):** {ssp_b_name}

**Statistics:**
- Controls added: {added_count}
- Controls removed: {removed_count}  
- Controls modified: {changed_count}
- Controls unchanged: {unchanged_count}

**Added Controls:**
{added_controls}

**Removed Controls:**
{removed_controls}

**Modified Controls:**
{changed_details}

Provide a security-focused summary of these changes.""",
        ),
    ]
)


def generate_diff_summary(diff: SSPDiff) -> str:
    """Generate an LLM-powered narrative summary of SSP differences."""
    llm = _get_llm()

    # Format changed controls details
    changed_details = []
    for ctrl_id, before, after in diff.changed_controls[:10]:  # Limit for context
        changed_details.append(
            f"**{ctrl_id}:**\n  Before: {before[:200]}...\n  After: {after[:200]}..."
            if len(before) > 200 or len(after) > 200
            else f"**{ctrl_id}:**\n  Before: {before}\n  After: {after}"
        )

    prompt_messages = _SUMMARY_PROMPT.format_messages(
        ssp_a_name=diff.ssp_a_name,
        ssp_b_name=diff.ssp_b_name,
        added_count=len(diff.added_controls),
        removed_count=len(diff.removed_controls),
        changed_count=len(diff.changed_controls),
        unchanged_count=len(diff.unchanged_controls),
        added_controls=", ".join(diff.added_controls) if diff.added_controls else "None",
        removed_controls=", ".join(diff.removed_controls) if diff.removed_controls else "None",
        changed_details="\n\n".join(changed_details) if changed_details else "None",
    )

    result = llm.invoke(prompt_messages)
    return result.content


def diff_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: compare two SSPs and summarize differences.

    Expects state with:
    - ssp_a_path: path to first SSP
    - ssp_b_path: path to second SSP

    Or parses paths from the last user message.
    """
    messages = state.get("messages", [])

    # Check if paths are directly in state
    ssp_a_path = state.get("ssp_a_path")
    ssp_b_path = state.get("ssp_b_path")

    if not ssp_a_path or not ssp_b_path:
        # Try to extract from last message
        if messages:
            last = messages[-1]
            if isinstance(last, HumanMessage):
                # Simple extraction - look for file paths
                content = last.content
                # This is a basic implementation - could be enhanced with NLP
                return {
                    "messages": [
                        AIMessage(
                            content="To compare SSPs, please provide both file paths. "
                            "Use the `compare_ssps()` function directly, or specify:\n"
                            "- ssp_a_path: path to the 'before' SSP\n"
                            "- ssp_b_path: path to the 'after' SSP"
                        )
                    ]
                }
        return {}

    try:
        diff = compare_ssps(ssp_a_path, ssp_b_path)
        summary = generate_diff_summary(diff)

        response = f"""## SSP Comparison Results

**Comparing:** {diff.ssp_a_name} → {diff.ssp_b_name}

### Statistics
- Added: {len(diff.added_controls)} controls
- Removed: {len(diff.removed_controls)} controls
- Modified: {len(diff.changed_controls)} controls
- Unchanged: {len(diff.unchanged_controls)} controls

### Summary
{summary}

### Details
**Added:** {', '.join(diff.added_controls) or 'None'}
**Removed:** {', '.join(diff.removed_controls) or 'None'}
**Changed:** {', '.join(c[0] for c in diff.changed_controls) or 'None'}
"""

        return {"messages": [AIMessage(content=response)], "diff_result": diff.to_dict()}

    except FileNotFoundError as e:
        return {"messages": [AIMessage(content=f"Error: Could not find SSP file - {e}")]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error comparing SSPs: {e}")]}
