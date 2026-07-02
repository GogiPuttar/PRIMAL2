from mapf_bench.core.problem import Action, Position


ACTION_TO_PRIMAL2 = {
    Action.WAIT: 0,
    Action.UP: 1,
    Action.DOWN: 2,
    Action.LEFT: 3,
    Action.RIGHT: 4,
}

PRIMAL2_TO_ACTION = {v: k for k, v in ACTION_TO_PRIMAL2.items()}


def grid_to_numpy(problem):
    import numpy as np

    grid = np.zeros((problem.grid.height, problem.grid.width), dtype=np.int8)
    for x, y in problem.grid.obstacles:
        grid[y, x] = 1
    return grid