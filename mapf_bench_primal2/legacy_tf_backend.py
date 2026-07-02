from __future__ import annotations

from typing import Any

from mapf_bench.plugins.base import StepRequest, StepResult


class LegacyTFBackend:
    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.checkpoint_path = params.get("checkpoint_path")
        if not self.checkpoint_path:
            raise ValueError("legacy_tf mode requires params.checkpoint_path")

    def reset(self, problem, *, seed=None) -> None:
        self.problem = problem
        self.seed = seed

        try:
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()
        except Exception as exc:
            raise RuntimeError(
                "PRIMAL2 legacy_tf mode requires a TensorFlow 1-compatible environment. "
                "Recommended: use a separate Python 3.7/3.8 conda or Docker environment "
                "for the PRIMAL2 adapter."
            ) from exc

    def step(self, request: StepRequest) -> StepResult:
        raise NotImplementedError(
            "TensorFlow import works, but PRIMAL2 model/session wiring is not implemented yet."
        )