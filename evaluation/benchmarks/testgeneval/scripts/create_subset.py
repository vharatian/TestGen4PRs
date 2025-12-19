"""
Create a subset of TDD-Bench dataset for testing.

Usage:
    python evaluation/benchmarks/testgeneval/scripts/create_subset.py \
        --input evaluation/benchmarks/testgeneval/data/tdd_bench.json \
        --output evaluation/benchmarks/testgeneval/data/tdd_bench_subset.json \
        --limit 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a subset of TDD-Bench dataset.")
    parser.add_argument("--input", required=True, help="Path to the source JSON file.")
    parser.add_argument("--output", required=True, help="Path to write the subset JSON file.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of instances to include (default: 10).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting index (default: 0).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"📖 Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"   Total instances: {len(data)}")

    # Create subset
    subset = data[args.offset : args.offset + args.limit]

    print(f"\n✂️  Creating subset:")
    print(f"   Offset: {args.offset}")
    print(f"   Limit: {args.limit}")
    print(f"   Selected: {len(subset)} instances")

    if subset:
        print(f"\n📋 First instance: {subset[0]['instance_id']}")
        print(f"   Last instance: {subset[-1]['instance_id']}")

    # Save subset
    print(f"\n💾 Saving subset to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(subset, f, indent=2)

    print(f"\n✅ Done! Created subset with {len(subset)} instances")
    print(f"\n💡 To use this subset, run:")
    print(f"   ./evaluation/benchmarks/testgeneval/scripts/run_infer.sh \\")
    print(f"     <model_config> HEAD CodeActAgent {len(subset)} 30 1 \\")
    print(f"     {output_path} \\")
    print(f"     test")


if __name__ == "__main__":
    main()
