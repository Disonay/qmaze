class State:
    def __init__(self, init_state: str, init_row: float, init_col: float):
        self.state = init_state,
        self.row = init_row
        self.col = init_col

    def set_valid(self):
        self.state = "valid"

    def set_blocked(self):
        self.state = "blocked"

    def set_invalid(self):
        self.state = "invalid"

    def is_win_state(self, target: tuple):
        return (self.row, self.col) == target

    def is_valid(self):
        return self.state == "valid"

    def is_blocked(self):
        return self.state == "blocked"

    def is_invalid(self):
        return self.state == "invalid"
