#!/usr/bin/env bash
set -euo pipefail

docker run --rm primal2:mapf-bench-legacy-tf \
  python mapf_bench_primal2/scripts/probe_legacy_network.py