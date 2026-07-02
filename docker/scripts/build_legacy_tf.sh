#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

docker build \
  -f docker/Dockerfile.mapf_bench_legacy_tf \
  -t primal2:mapf-bench-legacy-tf .