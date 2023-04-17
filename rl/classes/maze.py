import numpy as np

from rl.classes.action import Action
from rl.classes.state import State


class RLMaze:
    def __init__(self, maze, agent=(0, 0)):
        self.maze = None
        self.state = None
        self.min_reward = None
        self.total_reward = None
        self.visited = None

        self._maze = maze
        self.target = (maze.shape[0] - 1, maze.shape[1] - 1)
        self.free_cells = [
            (r, c)
            for r in range(self._maze.shape[0])
            for c in range(self._maze.shape[1])
            if self._maze[r, c] == 1.0 and (r, c) != self.target
        ]
        self.reset(agent)

    def reset(self, agent):
        self.maze = np.copy(self._maze)
        self.state = State("start", agent[0], agent[1])
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):

        if self.maze[self.state.row, self.state.col] > 0.0:
            self.visited.add(self.state.row, self.state.col)

        valid_actions = Action.valid_actions(self.state, self.maze)

        if not valid_actions:
            self.state.set_blocked()
        elif action in valid_actions:
            self.state.set_valid()
            if action == Action.LEFT:
                self.state.col -= 1
            elif action == Action.UP:
                self.state.row -= 1
            if action == Action.RIGHT:
                self.state.col += 1
            elif action == Action.DOWN:
                self.state.row += 1
        else:  # invalid action, no change in rat position
            self.state.set_invalid()

    def get_reward(self):
        if (self.state.row, self.state.col) == self.target:
            return 1.0
        if self.state.is_blocked():
            return self.min_reward - 1
        if (self.state.row, self.state.col) in self.visited:
            return -0.25
        if self.state.is_invalid():
            return -0.75
        if self.state.is_valid():
            return -0.04

    def get_envstate(self):
        envstate = self.maze.copy()
        envstate[self.state.row, self.state.col] = 0.5

        return envstate.reshape((1, -1))

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.get_envstate()

        return envstate, reward, status

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        if self.state.is_win_state(self.target):
            return 'win'

        return 'not_over'
