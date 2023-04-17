from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import PReLU
from rl.params import num_actions


def build_model(maze):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')

    return model
