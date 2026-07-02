#!/usr/bin/env bash
set -euo pipefail

docker run --rm primal2:mapf-bench-legacy-tf \
  bash -lc 'python - <<'"'"'PY'"'"'
import sys
from pathlib import Path

repo = Path("/workspace/PRIMAL2")
sys.path.insert(0, str(repo))

import tensorflow as tf
import ray

print("TensorFlow:", tf.__version__)
print("Ray:", ray.__version__)

import Ray_ACNet
import Primal2Observer
import Observer_Builder
import Map_Generator
import parameters

print("Core PRIMAL2 non-render imports OK")
PY'