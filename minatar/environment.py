################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
################################################################################################################
from importlib import import_module
import numpy as np
import cv2

from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sns

#####################################################################################################################
# Environment
#
# Wrapper for all the specific game environments. Imports the environment specified by the user and then acts as a
# minimal interface. Also defines code for displaying the environment for a human user. 
#
#####################################################################################################################
class Environment:
    def __init__(self, env_name, sticky_action_prob=0.1, difficulty_ramping=True, random_seed=None):
        env_module = import_module('minatar.environments.'+env_name)
        self.env_name = env_name
        self.env = env_module.Env(ramping=difficulty_ramping, seed=random_seed)
        self.env_state_shape = self.env.state_shape()
        self.n_channels = self.env_state_shape[2]
        self.sticky_action_prob = sticky_action_prob
        self.last_action = 0
        self.visualized = False
        self.closed = False

        self.cmap = sns.color_palette("cubehelix", self.n_channels)
        self.cmap_render = \
            [(np.uint8(255 * self.cmap[i][0]), np.uint8(255 * self.cmap[i][1]), np.uint8(255 * self.cmap[i][2]))
             for i in range(len(self.cmap))]

        self.cmap.insert(0, (0, 0, 0))
        self.cmap = colors.ListedColormap(self.cmap)
        bounds = [i for i in range(self.n_channels + 2)]
        self.norm = colors.BoundaryNorm(bounds, self.n_channels + 1)

    # Wrapper for env.act
    def act(self, a):
        if np.random.rand() < self.sticky_action_prob:
            a = self.last_action
        self.last_action = a
        return self.env.act(a)

    # Wrapper for env.state
    def state(self):
        return self.env.state()

    # Wrapper for env.reset
    def reset(self):
        return self.env.reset()

    # Wrapper for env.state_shape
    def state_shape(self):
        return self.env.state_shape()

    # All MinAtar environments have 6 actions
    def num_actions(self):
        return 6

    # Name of the MinAtar game associated with this environment
    def game_name(self):
        return self.env_name

    # Wrapper for env.minimal_action_set
    def minimal_action_set(self):
        return self.env.minimal_action_set()

    # Display the current environment state for time milliseconds using matplotlib
    def display_state(self, time=50):
        if not self.visualized:
            _, self.ax = plt.subplots(1, 1)
            plt.show(block=False)
            self.visualized = True

        if self.closed:
            _, self.ax = plt.subplots(1,1)
            plt.show(block=False)
            self.closed = False

        state = self.env.state()
        numerical_state = np.amax(state*np.reshape(np.arange(self.n_channels) + 1, (1, 1, -1)), 2) + 0.5

        plt.imshow(numerical_state, cmap=self.cmap, norm=self.norm, interpolation='none')

        plt.pause(time/1000)
        plt.cla()

    # Create rendered state observation
    def render_state(self, size=300):
        state = self.env.state()
        obs_image = np.zeros((self.env_state_shape[0], self.env_state_shape[1], 3), dtype=np.uint8)
        for x in range(self.env_state_shape[0]):
            for y in range(self.env_state_shape[1]):
                for z in range(self.env_state_shape[2]):
                    if state[x, y, z] == 1:
                        obs_image[x, y, :] = self.env_state_shape[z]
        obs_image = cv2.resize(obs_image, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        return obs_image

    def close_display(self):
        plt.close()
        self.closed = True
