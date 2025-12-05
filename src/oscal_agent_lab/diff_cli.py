#!/usr/bin/env python3
# src/oscal_agent_lab/diff_cli.py
"""CLI for comparing OSCAL SSPs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agents.diff import compare_ssps, generate_diff_summary


def main():
    """Run the SSP diff CLI."""
    parser = argparse.ArgumentParser(
        description="Compare two OSCAL SSPs and summarize changes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ssp_v1.json ssp_v2.json
  %(prog)s --no-summary ssp_before.json ssp_after.json
  %(prog)s --json ssp_a.json ssp_b.json > diff.json
""",
    )
    parser.add_argument("ssp_a", help="Path to the 'before' SSP (JSON)")
    parser.add_argument("ssp_b", help="Path to the 'after' SSP (JSON)")
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip LLM-generated summary (faster, no API call)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON diff instead of formatted text",
    )

    args = parser.parse_args()

    # Validate paths
    ssp_a_path = Path(args.ssp_a)
    ssp_b_path = Path(args.ssp_b)

    if not ssp_a_path.exists():
        print(f"Error: SSP file not found: {ssp_a_path}", file=sys.stderr)
        sys.exit(1)
    if not ssp_b_path.exists():
        print(f"Error: SSP file not found: {ssp_b_path}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Comparing SSPs...", file=sys.stderr)
        diff = compare_ssps(ssp_a_path, ssp_b_path)

        if args.json:
            import json

            result = diff.to_dict()
            if not args.no_summary:
                print("Generating summary...", file=sys.stderr)
                result["summary"] = generate_diff_summary(diff)
            print(json.dumps(result, indent=2))
        else:
            # Formatted output
            print(f"\n{'='*60}")
            print(f"SSP Comparison: {diff.ssp_a_name} → {diff.ssp_b_name}")
            print(f"{'='*60}\n")

            print(f"📊 Statistics:")
            print(f"   ✅ Added:     {len(diff.added_controls)} controls")
            print(f"   ❌ Removed:   {len(diff.removed_controls)} controls")
            print(f"   ✏️  Modified:  {len(diff.changed_controls)} controls")
            print(f"   ➖ Unchanged: {len(diff.unchanged_controls)} controls")
            print()

            if diff.added_controls:
                print(f"➕ Added Controls:")
                for ctrl in diff.added_controls:
                    print(f"   • {ctrl}")
                print()

            if diff.removed_controls:
                print(f"➖ Removed Controls:")
                for ctrl in diff.removed_controls:
                    print(f"   • {ctrl}")
                print()

            if diff.changed_controls:
                print(f"✏️  Modified Controls:")
                for ctrl_id, before, after in diff.changed_controls:
                    print(f"   • {ctrl_id}")
                print()

            if not args.no_summary:
                print("Generating AI summary...", file=sys.stderr)
                summary = generate_diff_summary(diff)
                print(f"{'─'*60}")
                print("📝 AI Summary:")
                print(f"{'─'*60}")
                print(summary)
                print()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
