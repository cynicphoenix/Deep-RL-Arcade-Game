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

class Agent(object):
    def __init__(self,
                 dqn,
                 target_dqn,
                 replay_buffer,
                 n_actions,
                 input_shape,
                 batch_size,
                 history_length=4,
                 eps_initial=1,
                 eps_final=0.1,
                 eps_final_frame=0.01,
                 eps_evaluation=0.0,
                 eps_annealing_frames=1000000,
                 replay_buffer_start_size=50000,
                 max_frames=25000000):
        self.replay_buffer_start_size = replay_buffer_start_size
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.history_length = history_length
        self.replay_buffer = replay_buffer
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.eps_initial = eps_initial
        self.eps_evaluation = eps_evaluation
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_annealing_frames = eps_annealing_frames
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope*self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (self.max_frames - self.eps_annealing_frames - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_frames
        self.DQN = dqn
        self.target_dqn = target_dqn

    def calc_epsilon(self, frame_number, evaluation=False):
        if evaluation:
            return self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            return self.eps_initial
        elif frame_number >= self.replay_buffer_start_size and frame_number < self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope*frame_number + self.intercept
        elif frame_number >= self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope_2*frame_number + self.intercept_2

    def get_action(self, frame_number, state, evaluation=False):
        eps = self.calc_epsilon(frame_number, evaluation)
        if np.random.rand(1) - eps < 0:
            return np.random.randint(0, self.n_actions)
        q_vals = self.DQN.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))[0].argmax()
        return q_vals

    def update_target_network(self):
        self.target_dqn.set_weights(self.DQN.get_weights())

    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        self.replay_buffer.add_experience(action, frame, reward, terminal, clip_reward)

    def learn(self, batch_size, gamma, frame_number, REPLAY_PRIORITY_SCALE=1.0):
        states, actions, rewards, new_states, terminal_flags = self.replay_buffer.get_minibatch(batch_size=self.batch_size, REPLAY_PRIORITY_SCALE=REPLAY_PRIORITY_SCALE)
        arg_q_max = self.DQN.predict(new_states).argmax(axis=1)
        future_q_vals = self.target_dqn.predict(new_states)
        double_q = future_q_vals[range(batch_size), arg_q_max]
        target_q = rewards + (gamma*double_q * (1-terminal_flags))
        with tf.GradientTape() as tape:
            q_values = self.DQN(states)
            one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions, dtype=np.float32)
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)
            error = Q - target_q
            loss = tf.keras.losses.Huber()(target_q, Q)
        model_gradients = tape.gradient(loss, self.DQN.trainable_variables)
        self.DQN.optimizer.apply_gradients(zip(model_gradients, self.DQN.trainable_variables))
        return float(loss.numpy()), error

    def save(self, folder_name, **kwargs):
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        self.DQN.save(folder_name + '/dqn.h5')
        self.target_dqn.save(folder_name + '/target_dqn.h5')
        self.replay_buffer.save(folder_name + '/replay-buffer')
        with open(folder_name + '/meta.json', 'w+') as f:
            f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current}, **kwargs}))

    def load(self, folder_name, load_replay_buffer=True):
        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')
        self.DQN = tf.keras.models.load_model(folder_name + '/dqn.h5')
        self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')
        self.optimizer = self.DQN.optimizer
        if load_replay_buffer:
            self.replay_buffer.load(folder_name + '/replay-buffer')
        with open(folder_name + '/meta.json', 'r') as f:
            meta = json.load(f)
        if load_replay_buffer:
            self.replay_buffer.count = meta['buff_count']
            self.replay_buffer.current = meta['buff_curr']
        del meta['buff_count'], meta['buff_curr']
        return meta