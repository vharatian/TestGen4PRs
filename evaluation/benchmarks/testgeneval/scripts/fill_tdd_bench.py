"""
Utility to normalize a TDD-Bench JSON/JSONL file into the TestGenEval schema.

It fills required fields that are missing by:
- duplicating `instance_id` into `id`
- wrapping issue context into `preds_context`
- providing empty defaults for `code_src`, `test_src`, `local_imports`, and `baseline_covs`
- optionally fetching code/test contents from GitHub

Usage:
    # Without fetching (fast, but code_src will be empty):
    python evaluation/benchmarks/testgeneval/scripts/fill_tdd_bench.py \
        --input evaluation/benchmarks/testgeneval/data/tdd_bench.json \
        --output evaluation/benchmarks/testgeneval/data/tdd_bench.filled.json

    # With GitHub fetching (slower, but populates code_src):
    python evaluation/benchmarks/testgeneval/scripts/fill_tdd_bench.py \
        --input evaluation/benchmarks/testgeneval/data/tdd_bench.json \
        --output evaluation/benchmarks/testgeneval/data/tdd_bench.filled.json \
        --fetch-code
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: 'requests' library not available. Use --fetch-code only if requests is installed.")

REQUIRED_FIELDS = {
    "repo",
    "base_commit",
    "version",
    "instance_id",
    "id",
    "patch",
    "test_patch",
    "preds_context",
    "code_src",
    "test_src",
    "code_file",
    "test_file",
    "local_imports",
    "baseline_covs",
}

DIFF_PATH_PATTERN = re.compile(r"^[+-]{3}\s+[ab]/(.+)")


def fetch_code_from_github(repo: str, commit: str, file_path: str) -> str:
    """
    Fetch file content from GitHub at specific commit.

    Args:
        repo: Repository in format "owner/repo"
        commit: Git commit hash
        file_path: Path to file in repository

    Returns:
        File content as string, or empty string if fetch fails
    """
    if not REQUESTS_AVAILABLE:
        return ""

    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{file_path}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            print(f"  ⚠️  Failed to fetch {file_path} (status {response.status_code})")
            return ""
    except Exception as e:
        print(f"  ⚠️  Error fetching {file_path}: {e}")
        return ""


def _extract_paths_from_diff(diff_text: str) -> list[str]:
    """Return file paths mentioned in a unified diff header."""
    paths: list[str] = []
    for line in diff_text.splitlines():
        m = DIFF_PATH_PATTERN.match(line)
        if m:
            paths.append(m.group(1).strip())
    return paths


def _guess_code_and_test_paths(patch: str, test_patch: str) -> tuple[str, str]:
    """
    Best-effort guess of code_file and test_file from diffs.
    - Prefer first non-test path for code_file.
    - Prefer first test-like path for test_file; fall back to patch paths.
    """
    code_file = ""
    test_file = ""

    # Paths from main patch
    patch_paths = _extract_paths_from_diff(patch)
    # Paths from test patch
    test_paths = _extract_paths_from_diff(test_patch)

    def is_test_path(p: str) -> bool:
        return "test" in p.lower() or p.startswith("tests/") or p.endswith("_test.py")

    # Guess test_file first from test_patch paths
    for p in test_paths:
        if is_test_path(p):
            test_file = p
            break
    if not test_file:
        for p in patch_paths:
            if is_test_path(p):
                test_file = p
                break

    # Guess code_file from non-test paths in patch
    for p in patch_paths:
        if not is_test_path(p):
            code_file = p
            break

    return code_file, test_file


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    text_stripped = text.strip()
    if not text_stripped:
        return []
    if text_stripped.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_record(
    raw: dict[str, Any],
    missing_paths: list[str],
    fetch_code: bool = False,
    stats: dict[str, int] | None = None
) -> dict[str, Any]:
    """
    Normalize a single TDD-Bench record into TestGenEval schema.

    Args:
        raw: Raw record from TDD-Bench
        missing_paths: List to append missing path warnings to
        fetch_code: If True, fetch code_src and test_src from GitHub
        stats: Dictionary to track statistics (optional)

    Returns:
        Normalized record
    """
    out = dict(raw)

    # Ensure id duplication
    out.setdefault("id", out.get("instance_id", ""))

    # Ensure preds_context exists and absorbs issue/PR hints if present
    preds_context = out.get("preds_context") or {}
    if "problem_statement" in out:
        preds_context.setdefault("problem_statement", out["problem_statement"])
    if "hints_text" in out:
        preds_context.setdefault("hints_text", out["hints_text"])
    if "difficulty" in out:
        preds_context.setdefault("difficulty", out["difficulty"])
    if "created_at" in out:
        preds_context.setdefault("created_at", out["created_at"])
    if "environment_setup_commit" in out:
        preds_context.setdefault("environment_setup_commit", out["environment_setup_commit"])
    out["preds_context"] = preds_context

    # Provide safe defaults
    out.setdefault("code_src", "")
    out.setdefault("test_src", "")
    out.setdefault("local_imports", [])
    out.setdefault("baseline_covs", {})

    # Try to infer paths from diffs if missing
    if not out.get("code_file") or not out.get("test_file"):
        code_file_guess, test_file_guess = _guess_code_and_test_paths(
            out.get("patch", "") or "", out.get("test_patch", "") or ""
        )
        out.setdefault("code_file", code_file_guess or "")
        out.setdefault("test_file", test_file_guess or "")

    # Track still-missing paths
    if not out.get("code_file"):
        missing_paths.append(out.get("instance_id", "<unknown>") + " missing code_file")
    if not out.get("test_file"):
        missing_paths.append(out.get("instance_id", "<unknown>") + " missing test_file")

    # Fetch code from GitHub if requested
    if fetch_code and out.get("repo") and out.get("base_commit"):
        # Fetch code_src if not already present and we have a code_file
        if not out.get("code_src") and out.get("code_file"):
            code_src = fetch_code_from_github(
                out["repo"],
                out["base_commit"],
                out["code_file"]
            )
            if code_src:
                out["code_src"] = code_src
                if stats:
                    stats["code_fetched"] += 1
            else:
                if stats:
                    stats["code_fetch_failed"] += 1
            # Rate limiting to be nice to GitHub
            time.sleep(0.1)

        # Fetch test_src if not already present and we have a test_file
        if not out.get("test_src") and out.get("test_file"):
            test_src = fetch_code_from_github(
                out["repo"],
                out["base_commit"],
                out["test_file"]
            )
            if test_src:
                out["test_src"] = test_src
                if stats:
                    stats["test_fetched"] += 1
            else:
                if stats:
                    stats["test_fetch_failed"] += 1
            # Rate limiting
            time.sleep(0.1)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize TDD-Bench JSON into TestGenEval schema.")
    parser.add_argument("--input", required=True, help="Path to the source JSON/JSONL file.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the normalized JSON file (array).",
    )
    parser.add_argument(
        "--fetch-code",
        action="store_true",
        help="Fetch code_src and test_src from GitHub (requires 'requests' library).",
    )
    args = parser.parse_args()

    if args.fetch_code and not REQUESTS_AVAILABLE:
        print("❌ Error: --fetch-code requires the 'requests' library.")
        print("   Install it with: pip install requests")
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading data from {input_path}...")
    data = load_json_or_jsonl(input_path)
    print(f"Loaded {len(data)} records")

    if args.fetch_code:
        print("\n🌐 Fetching code from GitHub (this may take a while)...")
        print("   Progress will be shown below:")

    missing_paths: list[str] = []
    stats = {
        "code_fetched": 0,
        "code_fetch_failed": 0,
        "test_fetched": 0,
        "test_fetch_failed": 0,
    }

    normalized = []
    for i, rec in enumerate(data, 1):
        if args.fetch_code and i % 10 == 0:
            print(f"   Processing {i}/{len(data)}...")
        normalized_rec = normalize_record(rec, missing_paths, fetch_code=args.fetch_code, stats=stats)
        normalized.append(normalized_rec)

    # Save as a JSON array for clarity
    print(f"\n💾 Saving normalized data to {output_path}...")
    output_path.write_text(json.dumps(normalized, indent=2))

    # Print summary
    print(f"\n✅ Normalized {len(normalized)} records to {output_path}")

    if args.fetch_code:
        print("\n📊 GitHub Fetch Statistics:")
        print(f"   Code files fetched: {stats['code_fetched']}")
        print(f"   Code files failed: {stats['code_fetch_failed']}")
        print(f"   Test files fetched: {stats['test_fetched']}")
        print(f"   Test files failed: {stats['test_fetch_failed']}")

    if missing_paths:
        print(f"\n⚠️  {len(missing_paths)} warnings about missing path fields:")
        for note in missing_paths[:10]:  # Show first 10
            print(f"   - {note}")
        if len(missing_paths) > 10:
            print(f"   ... and {len(missing_paths) - 10} more")

    # Validation check
    print("\n🔍 Validation Check:")
    issues = []
    for rec in normalized:
        if not rec.get('code_src'):
            issues.append(f"{rec['instance_id']}: Missing code_src")
        if not rec.get('code_file'):
            issues.append(f"{rec['instance_id']}: Missing code_file")
        if not rec.get('test_file'):
            issues.append(f"{rec['instance_id']}: Missing test_file")

    if issues:
        print(f"   ⚠️  {len(issues)} critical issues found:")
        for issue in issues[:10]:
            print(f"      - {issue}")
        if len(issues) > 10:
            print(f"      ... and {len(issues) - 10} more")
        print("\n   ⚠️  These instances may FAIL during evaluation!")
        if not args.fetch_code:
            print("   💡 Consider using --fetch-code to populate code_src from GitHub")
    else:
        print("   ✅ All instances have required fields!")


if __name__ == "__main__":
    main()
