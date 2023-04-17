import numpy as np


def play_game(model, qmaze, rat_cell):
    qmaze.reset(rat_cell)
    envstate = qmaze.get_envstate()
    actions = []
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])
        actions.append((qmaze.state["position_row"], qmaze.state["position_col"]))
        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            actions.append((qmaze.state["position_row"], qmaze.state["position_col"]))
            return actions
        elif game_status == 'lose':
            actions.append((qmaze.state["position_row"], qmaze.state["position_col"]))
            return actions