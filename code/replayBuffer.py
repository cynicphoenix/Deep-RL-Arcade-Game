from config import *
import numpy as np
import cv2
import random
import os
import json
import time
import gym
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

class ReplayBuffer:
    def __init__(self, size=1000000, input_shape=(84, 84), history_length=4):
        self.size = size
        self.input_shape = input_shape
        self.history_length = history_length
        self.count = 0 
        self.current = 0
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        if clip_reward:
            reward = np.sign(reward)
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def get_minibatch(self, batch_size=32, REPLAY_PRIORITY_SCALE=0.0):
        if self.count < self.history_length:
            raise ValueError('Not enough exp.')
        indices = []
        for i in range(batch_size):
            while True:
                index = random.randint(self.history_length, self.count - 1)
                if (index >= self.current and index - self.history_length <= self.current) or (self.terminal_flags[index - self.history_length:index].any()):
                    continue
                break
            indices.append(index)
        states = []
        new_states = []
        for idx in indices:
            states.append(self.frames[idx-self.history_length:idx, ...])
            new_states.append(self.frames[idx-self.history_length+1:idx+1, ...])
        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))
        return states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]

    def save(self, folder_name):
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/frames.npy', self.frames)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/terminal_flags.npy', self.terminal_flags)

    def load(self, folder_name):
        self.actions = np.load(folder_name + '/actions.npy')
        self.frames = np.load(folder_name + '/frames.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.terminal_flags = np.load(folder_name + '/terminal_flags.npy')