# src/oscal_agent_lab/oscal_loader.py
"""Load and flatten OSCAL JSON into LangChain Documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def load_oscal_json(path: str | Path) -> dict[str, Any]:
    """Load an OSCAL JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def catalog_to_documents(catalog_json: dict[str, Any]) -> list[Document]:
    """
    Flatten an OSCAL catalog into Documents (roughly one per control).

    This assumes the typical OSCAL catalog layout:
      { "catalog": { "groups": [ { "id": ..., "controls": [ ... ] }, ... ] } }
    as used in the NIST SP 800-53 Rev 5 OSCAL examples.
    """
    catalog = catalog_json.get("catalog", catalog_json)
    docs: list[Document] = []

    groups = catalog.get("groups", [])
    for group in groups:
        group_id = group.get("id")
        group_title = group.get("title", "")
        _process_controls(group.get("controls", []), group_id, group_title, docs)

    return docs


def _process_controls(
    controls: list[dict[str, Any]],
    group_id: str | None,
    group_title: str,
    docs: list[Document],
    parent_id: str | None = None,
) -> None:
    """Recursively process controls and their enhancements."""
    for control in controls:
        control_id = control.get("id")
        title = control.get("title", "")
        parts = control.get("parts", [])
        props = control.get("props", [])

        text_parts: list[str] = []

        # Control header
        if title:
            if parent_id:
                text_parts.append(f"Control Enhancement {control_id}: {title}")
                text_parts.append(f"(Enhancement of {parent_id})")
            else:
                text_parts.append(f"Control {control_id}: {title}")

        # Add group context
        if group_title:
            text_parts.append(f"Family: {group_title}")

        # Extract properties (like control status, labels)
        for prop in props:
            if prop.get("name") == "label":
                text_parts.append(f"Label: {prop.get('value', '')}")

        # Extract prose from parts (statement, guidance, etc.)
        _extract_parts(parts, text_parts)

        if not text_parts:
            continue

        content = "\n\n".join(text_parts)
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "control_id": control_id,
                    "group_id": group_id,
                    "group_title": group_title,
                    "parent_control": parent_id,
                    "source": "oscal_catalog",
                },
            )
        )

        # Process control enhancements (nested controls)
        enhancements = control.get("controls", [])
        if enhancements:
            _process_controls(enhancements, group_id, group_title, docs, parent_id=control_id)


def _extract_parts(parts: list[dict[str, Any]], text_parts: list[str], indent: int = 0) -> None:
    """Recursively extract prose from OSCAL parts."""
    for part in parts:
        name = part.get("name") or part.get("title") or ""
        prose = part.get("prose", "")

        prefix = "  " * indent
        if prose:
            if name:
                text_parts.append(f"{prefix}{name.capitalize()}: {prose}")
            else:
                text_parts.append(f"{prefix}{prose}")

        # Handle nested parts (e.g., statement items a, b, c)
        nested_parts = part.get("parts", [])
        if nested_parts:
            _extract_parts(nested_parts, text_parts, indent + 1)


def ssp_to_documents(ssp_json: dict[str, Any]) -> list[Document]:
    """
    Flatten an OSCAL SSP into Documents.

    Extracts:
    - System metadata
    - Implemented requirements (per control)
    - Component information
    """
    ssp = ssp_json.get("system-security-plan", ssp_json)
    docs: list[Document] = []

    # System metadata
    metadata = ssp.get("metadata", {})
    system_name = metadata.get("title", "Unknown System")

    # System characteristics
    sys_chars = ssp.get("system-characteristics", {})
    sys_id = sys_chars.get("system-id", [{}])[0].get("id", "") if sys_chars.get("system-id") else ""

    # Control implementations
    control_impl = ssp.get("control-implementation", {})
    impl_reqs = control_impl.get("implemented-requirements", [])

    for req in impl_reqs:
        control_id = req.get("control-id", "")
        statements = req.get("statements", [])
        by_components = req.get("by-components", [])

        text_parts: list[str] = [f"System: {system_name}", f"Control: {control_id}"]

        # Extract statement-level implementations
        for stmt in statements:
            stmt_id = stmt.get("statement-id", "")
            text_parts.append(f"\nStatement {stmt_id}:")
            for comp in stmt.get("by-components", []):
                desc = comp.get("description", "")
                if desc:
                    text_parts.append(f"  Implementation: {desc}")

        # Extract control-level by-component implementations
        for comp in by_components:
            desc = comp.get("description", "")
            if desc:
                text_parts.append(f"Implementation: {desc}")

        if len(text_parts) > 2:  # More than just system name and control id
            docs.append(
                Document(
                    page_content="\n".join(text_parts),
                    metadata={
                        "control_id": control_id,
                        "system_name": system_name,
                        "system_id": sys_id,
                        "source": "oscal_ssp",
                    },
                )
            )

    return docs


def profile_to_documents(profile_json: dict[str, Any]) -> list[Document]:
    """
    Flatten an OSCAL profile into Documents.

    Extracts:
    - Profile metadata
    - Imported controls and modifications
    """
    profile = profile_json.get("profile", profile_json)
    docs: list[Document] = []

    metadata = profile.get("metadata", {})
    profile_title = metadata.get("title", "Unknown Profile")

    # Imports (which controls are selected)
    imports = profile.get("imports", [])

    for imp in imports:
        href = imp.get("href", "")
        include_controls = imp.get("include-controls", [])

        for inc in include_controls:
            with_ids = inc.get("with-ids", [])
            for control_id in with_ids:
                docs.append(
                    Document(
                        page_content=f"Profile: {profile_title}\nIncludes control: {control_id}\nSource: {href}",
                        metadata={
                            "control_id": control_id,
                            "profile_title": profile_title,
                            "source_href": href,
                            "source": "oscal_profile",
                        },
                    )
                )

    return docs
