#!/usr/bin/env python3
# src/oscal_agent_lab/cli.py
"""Simple CLI REPL for OSCAL Agent Lab."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage

from .graph import build_graph


def print_banner():
    """Print the CLI banner."""
    print(
        """
╔═══════════════════════════════════════════════════════════╗
║              OSCAL Agent Lab – v0.4                       ║
║                                                           ║
║  Agents: Explainer • Diff • ProfileBuilder • Validator    ║
║  Type 'help' for commands, 'exit' to leave.               ║
╚═══════════════════════════════════════════════════════════╝
"""
    )


def main():
    """Run the CLI REPL."""
    print_banner()

    try:
        graph = build_graph()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nSetup instructions:")
        print("  1. cd to your project root")
        print("  2. git submodule add https://github.com/usnistgov/oscal-content.git data/oscal-content")
        print("  3. Set OPENAI_API_KEY environment variable")
        print("  4. Run again: python -m oscal_agent_lab.cli")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        sys.exit(1)

    state = {"messages": []}

    print("Ready! Ask me about OSCAL controls.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye! 👋")
            break

        # Special commands
        if user_input.lower() == "clear":
            state = {"messages": []}
            print("Conversation cleared.\n")
            continue

        if user_input.lower() == "help":
            print(
                """
Available commands:
  help              - Show this help message
  clear             - Clear conversation history
  diff <a> <b>      - Compare two SSP files
  exit              - Exit the CLI

Example questions:
  - What does AC-2 require?
  - Explain the difference between AU-2 and AU-3
  - Which controls cover password policies?
  - What are the key requirements in the SI family?

Example diff:
  diff data/test-ssps/ssp_v1.json data/test-ssps/ssp_v2.json

Profile building:
  profile <description>   - Generate OSCAL profile from system description
  profile --baseline low "My simple web app"

Validation:
  validate <file>         - Validate OSCAL JSON against schema
"""
            )
            continue

        # Handle validate command
        if user_input.lower().startswith("validate "):
            from .agents.validator import validate_file, explain_errors
            file_path = user_input[9:].strip()

            if not file_path:
                print("Usage: validate <oscal_file.json>\n")
                continue

            if not Path(file_path).exists():
                print(f"Error: File not found: {file_path}\n")
                continue

            try:
                print(f"Validating {file_path}...")
                result = validate_file(file_path)

                if result.valid:
                    print(f"\n✅ Valid OSCAL {result.model_type} document!\n")
                else:
                    print(f"\n❌ Validation failed: {result.error_count} errors")
                    print(f"   Model type: {result.model_type}\n")

                    for i, err in enumerate(result.errors[:5], 1):
                        print(f"   {i}. {err.path}: {err.message[:80]}...")

                    if result.error_count > 5:
                        print(f"   ... and {result.error_count - 5} more errors\n")

                    print("\nGenerating fix suggestions...")
                    suggestions = explain_errors(result)
                    print(f"\n📝 How to fix:\n{suggestions}\n")

            except Exception as e:
                print(f"Error: {e}\n")
            continue

        # Handle profile command
        if user_input.lower().startswith("profile "):
            from .agents.profile_builder import build_profile
            import json

            # Parse arguments
            args = user_input[8:].strip()
            baseline = "moderate"

            # Check for --baseline flag
            if args.startswith("--baseline "):
                parts = args.split(maxsplit=2)
                if len(parts) >= 2:
                    baseline = parts[1].lower()
                    args = parts[2] if len(parts) > 2 else ""

            if not args:
                print("Usage: profile [--baseline low|moderate|high] <system description>\n")
                continue

            try:
                print(f"Building profile (baseline: {baseline})...")
                result = build_profile(args, baseline=baseline)

                print(f"\n📋 Profile Generated")
                print(f"   Baseline: {baseline} ({result['baseline_count']} controls)")
                print(f"   Total: {result['control_count']} controls")
                print(f"   Added: {result['added_count']} | Removed: {result['removed_count']}")
                print(f"\n📝 Rationale:")
                print(f"   {result['analysis'].get('rationale', 'N/A')}")

                if result['analysis'].get('additional_controls'):
                    print(f"\n➕ Additional: {', '.join(result['analysis']['additional_controls'])}")
                if result['analysis'].get('removed_controls'):
                    print(f"➖ Removed: {', '.join(result['analysis']['removed_controls'])}")

                # Save profile to file
                output_file = "generated_profile.json"
                with open(output_file, "w") as f:
                    json.dump(result['profile'], f, indent=2)
                print(f"\n✅ Profile saved to: {output_file}\n")

            except Exception as e:
                print(f"Error: {e}\n")
            continue

        # Handle diff command
        if user_input.lower().startswith("diff "):
            parts = user_input.split(maxsplit=2)
            if len(parts) != 3:
                print("Usage: diff <ssp_a_path> <ssp_b_path>\n")
                continue

            from .agents.diff import compare_ssps, generate_diff_summary
            from pathlib import Path

            ssp_a, ssp_b = parts[1], parts[2]
            if not Path(ssp_a).exists():
                print(f"Error: File not found: {ssp_a}\n")
                continue
            if not Path(ssp_b).exists():
                print(f"Error: File not found: {ssp_b}\n")
                continue

            try:
                print("Comparing SSPs...")
                diff = compare_ssps(ssp_a, ssp_b)
                print(f"\n📊 {diff.ssp_a_name} → {diff.ssp_b_name}")
                print(f"   Added: {len(diff.added_controls)} | Removed: {len(diff.removed_controls)} | Changed: {len(diff.changed_controls)}")
                if diff.added_controls:
                    print(f"   ➕ {', '.join(diff.added_controls)}")
                if diff.removed_controls:
                    print(f"   ➖ {', '.join(diff.removed_controls)}")
                if diff.changed_controls:
                    print(f"   ✏️  {', '.join(c[0] for c in diff.changed_controls)}")
                print("\nGenerating summary...")
                summary = generate_diff_summary(diff)
                print(f"\n{summary}\n")
            except Exception as e:
                print(f"Error: {e}\n")
            continue

        # Add user message and invoke graph
        state["messages"].append(HumanMessage(content=user_input))

        try:
            state = graph.invoke(state)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue

        # Extract and print the last AI message
        ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            print(f"\nAgent: {ai_messages[-1].content}\n")
        else:
            print("\n(No response generated)\n")


if __name__ == "__main__":
    main()
