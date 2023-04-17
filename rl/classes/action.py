from rl.classes.state import State
from rl.params import LEFT, UP, RIGHT, DOWN
from enum import Enum


class Action(Enum):
    LEFT: 0
    UP: 1
    RIGHT: 2
    DOWN: 3

    @staticmethod
    def valid_actions(state: State, maze, cell=None):
        if cell is None:
            row, col, mode = (state.row, state.col, state.state)
        else:
            row, col = cell

        actions = [Action.LEFT, Action.UP, Action.RIGHT, Action.DOWN]

        n_rows, n_cols = maze.shape
        if row == 0:
            actions.remove(Action.UP)
        elif row == n_rows - 1:
            actions.remove(Action.DOWN)

        if col == 0:
            actions.remove(Action.LEFT)
        elif col == n_cols - 1:
            actions.remove(Action.RIGHT)

        if row > 0 and maze[row - 1, col] == 0.0:
            actions.remove(Action.UP)
        if row < n_rows - 1 and maze[row + 1, col] == 0.0:
            actions.remove(Action.DOWN)

        if col > 0 and maze[row, col - 1] == 0.0:
            actions.remove(Action.LEFT)
        if col < n_cols - 1 and maze[row, col + 1] == 0.0:
            actions.remove(Action.RIGHT)

        return actions
