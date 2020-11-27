import gym
import random
import numpy as np
from train_dqn import *

class AtariWrapper:
    def __init__(self, ATARI_ENV, no_op_steps=10, history_length=4):
        self.env = gym.make(ATARI_ENV)
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.history_length = history_length

    def reset(self, evaluation=False):
        self.last_lives = 0
        self.frame = self.env.reset()
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.env.step(1)
        self.state = np.repeat(resize_frame(self.frame), self.history_length, axis=2)

    def step(self, action, render_mode=None):
        new_frame, reward, terminal, info = self.env.step(action)
        life_lost = True if (info['ale.lives'] < self.last_lives) else terminal
        self.last_lives = info['ale.lives']
        processed_frame = resize_frame(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)
        if render_mode == 'rgb_array':
            return processed_frame, reward, terminal, life_lost, self.env.render(render_mode)
        elif render_mode == 'human':
            self.env.render()
        return processed_frame, reward, terminal, life_lost