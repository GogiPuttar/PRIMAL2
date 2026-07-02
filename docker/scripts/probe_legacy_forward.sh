#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker run --rm \
  -v "$REPO_ROOT:/workspace/PRIMAL2" \
  -w /workspace/PRIMAL2 \
  primal2:mapf-bench-legacy-tf \
  python mapf_bench_primal2/scripts/probe_legacy_forward.py