import datetime
import sys

from rl.classes.maze import RLMaze
from rl.classes.experience import Experience
from rl.params import epsilon
import random
import numpy as np


def q_train(model, maze, opt):
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    start_time = datetime.datetime.now()

    # Construct environment/game from numpy array: maze (see above)
    rlmaze = RLMaze(maze)

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)

    win_history = []  # history of win/lose game
    hsize = rlmaze.maze.size // 2  # history window size
    win_rate = 0.0
    with open("log.txt", 'w') as log:
        sys.stdout = log
        for epoch in range(n_epoch):
            loss = 0.0
            agent_cell = random.choice(rlmaze.free_cells)
            rlmaze.reset(agent_cell)
            game_over = False

            envstate = rlmaze.get_envstate()
            n_episodes = 0
            while not game_over:

                valid_actions = rlmaze.valid_actions()
                if not valid_actions: break

                prev_envstate = envstate
                # Get next action
                if np.random.rand() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    action = np.argmax(experience.predict(prev_envstate))

                # Apply action, get reward and new envstate
                envstate, reward, game_status = rlmaze.act(action)
                if game_status == 'win':
                    win_history.append(1)
                    game_over = True
                elif game_status == 'lose':
                    win_history.append(0)
                    game_over = True
                else:
                    game_over = False

                # Store episode (experience)
                episode = [prev_envstate, action, reward, envstate, game_over]
                experience.remember(episode)
                n_episodes += 1

                # Train neural network model
                inputs, targets = experience.get_data(data_size=data_size)
                model.fit(
                    inputs,
                    targets,
                    epochs=8,
                    batch_size=16,
                    verbose=0,
                )
                loss = model.evaluate(inputs, targets, verbose=0)

            if len(win_history) > hsize:
                win_rate = sum(win_history[-hsize:]) / hsize

            dt = datetime.datetime.now() - start_time
            template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | time: {}"
            print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), dt.total_seconds()))