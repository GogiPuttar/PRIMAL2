from __future__ import annotations

from typing import Any, Mapping

from mapf_bench.core.problem import Action, MAPFProblem, Position
from mapf_bench.plugins.base import (
    PathfinderCapabilities,
    PlanRequest,
    PlanResult,
    StepRequest,
    StepResult,
)


class PRIMAL2Pathfinder:
    plugin_id = "marmotlab/primal2"

    capabilities = PathfinderCapabilities(
        supports_step=True,
        supports_full_plan=False,
        supports_lifelong=True,
        supports_training=False,
        centralized=False,
        decentralized=True,
        requires_gpu=False,
        requires_compiled_extensions=False,
    )

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.problem: MAPFProblem | None = None
        self.goals: dict[str, Position] = {}
        self.agent_ids: list[str] = []
        self.checkpoint_path: str | None = None
        self.mode = "compat_greedy"

    def configure(self, config):
        self.params = dict(config)
        self.mode = self.params.get("mode", "compat_greedy")

        if self.mode == "legacy_tf":
            from mapf_bench_primal2.legacy_tf_backend import LegacyTFBackend
            self.backend = LegacyTFBackend(self.params)
        else:
            self.backend = None

    def reset(self, problem: MAPFProblem, *, seed: int | None = None) -> None:
        self.problem = problem
        self.goals = {a.agent_id: a.goal for a in problem.agents}
        self.agent_ids = [a.agent_id for a in problem.agents]
        if self.backend is not None:
            self.backend.reset(problem, seed=seed)

    def step(self, request: StepRequest) -> StepResult:
        if self.problem is None:
            self.reset(request.problem, seed=request.seed)

        if self.backend is not None:
            return self.backend.step(request)

        actions: dict[str, Action] = {}

        for agent_id in self.agent_ids:
            current = request.positions[agent_id]
            goal = self.goals[agent_id]
            actions[agent_id] = self._greedy_action(current, goal)

        return StepResult(
            actions=actions,
            metadata={
                "adapter": self.plugin_id,
                "mode": self.mode,
                "note": "compat_greedy placeholder for PRIMAL2 plugin integration",
            },
        )

    def plan(self, request: PlanRequest) -> PlanResult:
        return PlanResult(
            status="unsupported",
            message="PRIMAL2 adapter currently exposes decentralized step() only.",
        )

    def close(self) -> None:
        self.problem = None

    def _greedy_action(self, current: Position, goal: Position) -> Action:
        assert self.problem is not None

        candidates = sorted(
            list(Action),
            key=lambda action: self._manhattan(
                self._apply_action(current, action),
                goal,
            ),
        )

        for action in candidates:
            proposed = self._apply_action(current, action)
            if self.problem.grid.is_free(proposed):
                return action

        return Action.WAIT

    @staticmethod
    def _apply_action(pos: Position, action: Action) -> Position:
        if action == Action.WAIT:
            return pos
        if action == Action.UP:
            return pos[0], pos[1] - 1
        if action == Action.DOWN:
            return pos[0], pos[1] + 1
        if action == Action.LEFT:
            return pos[0] - 1, pos[1]
        if action == Action.RIGHT:
            return pos[0] + 1, pos[1]
        raise ValueError(f"Unsupported action: {action}")

    @staticmethod
    def _manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])