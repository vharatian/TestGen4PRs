# Local dataset drop zone

Place your local TestGenEval-compatible datasets here, e.g. `tdd_bench.json` or `tdd_bench.jsonl`.
- Supported formats: JSON array or JSONL (one JSON object per line).
- Required keys per instance: `repo`, `base_commit`, `version`, `instance_id`, `id`, `patch`, `test_patch`, `preds_context`, `code_src`, `test_src`, `code_file`, `test_file`, `local_imports`, `baseline_covs`.
- Point the runners at the file with `--dataset evaluation/benchmarks/testgeneval/data/tdd_bench.json --split test`.
