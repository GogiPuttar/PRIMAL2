#!/usr/bin/env python3

import inspect
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))

import tensorflow as tf

import Ray_ACNet


def main():
    print("TensorFlow:", tf.__version__)
    print("ACNet class:", Ray_ACNet.ACNet)

    print("\nACNet.__init__ signature:")
    print(inspect.signature(Ray_ACNet.ACNet.__init__))

    print("\nACNet methods/attributes:")
    for name, value in inspect.getmembers(Ray_ACNet.ACNet):
        if not name.startswith("_"):
            kind = "method" if inspect.isfunction(value) else type(value).__name__
            print(f"  {name}: {kind}")

    print("\nRay_ACNet constants:")
    for name in [
        "RNN_SIZE",
        "GOAL_REPR_SIZE",
        "KEEP_PROB1",
        "KEEP_PROB2",
        "GRAD_CLIP",
    ]:
        if hasattr(Ray_ACNet, name):
            print(f"  {name} = {getattr(Ray_ACNet, name)}")


if __name__ == "__main__":
    main()