import matplotlib.pyplot as plt
import numpy as np
import os, time, sys
import tensorflow as tf
from config import *
from train_dqn import gen_qnetwork, resize_frame
from agent_dueling_dqn import Agent
from AtariWrapper import AtariWrapper
from replayBuffer import ReplayBuffer

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    SAVED_MODEL_LOC = sys.argv[1]
    if SAVED_MODEL_LOC is None:
        sys.exit(0)
    TARGET_DQN = gen_qnetwork(atari_wrap.env.action_space.n, input_shape=INPUT_SHAPE)
    MAIN_DQN = gen_qnetwork(atari_wrap.env.action_space.n, LEARNING_RATE_ALPHA, input_shape=INPUT_SHAPE)
    atari_wrap = AtariWrapper(ATARI_ENV, MAX_NOOP_STEPS)
    replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE)
    agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, atari_wrap.env.action_space.n, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE)
    agent.load(SAVED_MODEL_LOC)
    terminal = True
    eval_rewards = []
    for frame in range(EVAL_LENGTH):
        if terminal:
            atari_wrap.reset(evaluation=True)
            life_lost = True
            episode_reward_sum = 0
            terminal = False
        action = 1 if life_lost else agent.get_action(0, atari_wrap.state, evaluation=True) #Breakout Specific Instruction
        _, reward, terminal, life_lost = atari_wrap.step(action, render_mode='human')
        episode_reward_sum += reward
        if terminal:
            print(f'Reward: {episode_reward_sum}, frame: {frame}/{EVAL_LENGTH}')
            eval_rewards.append(episode_reward_sum)
    if len(eval_rewards) > 0:
        print('Avg Reward:', np.mean(eval_rewards))