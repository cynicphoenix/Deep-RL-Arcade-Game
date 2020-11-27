from config import *
import numpy as np
import cv2
import random
import os
import json
import time
import gym
import sys
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from agent_dueling_dqn import *
from replayBuffer import *
from AtariWrapper import *

def resize_frame(frame, shape=(84, 84)):
    frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)[34:34+160, :160]
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    return frame.reshape((*shape, 1))

def gen_qnetwork(n_actions, LEARNING_RATE_ALPHA=0.00001, input_shape=(84, 84), history_length=4):
    model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
    x = Lambda(lambda layer: layer / 255)(model_input)
    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)
    val_stream = Flatten()(val_stream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)
    adv_stream = Flatten()(adv_stream)
    adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)
    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))
    q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])
    model = Model(model_input, q_vals)
    model.compile(Adam(LEARNING_RATE_ALPHA), loss=tf.keras.losses.Huber())
    return model

if __name__ == "__main__":
    if len(sys.argv) > 1:
        SAVE_LOC_LAST_MODEL = sys.argv[1]
    atari_wrap = AtariWrapper(ATARI_ENV, MAX_NOOP_STEPS)
    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
    MAIN_DQN = gen_qnetwork(atari_wrap.env.action_space.n, LEARNING_RATE_ALPHA, input_shape=INPUT_SHAPE)
    TARGET_DQN = gen_qnetwork(atari_wrap.env.action_space.n, input_shape=INPUT_SHAPE)
    replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE)
    agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, atari_wrap.env.action_space.n, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE)
    if SAVE_LOC_LAST_MODEL is None:
        frame_number = 0
        rewards = []
        loss_list = []
    else:
        print('Loading Model', SAVE_LOC_LAST_MODEL)
        meta = agent.load(SAVE_LOC_LAST_MODEL, LOAD_REPLAY_BUFFER)
        frame_number = meta['frame_number']
        rewards = meta['rewards']
        loss_list = meta['loss_list']
    try:
        with writer.as_default():
            while frame_number < TOTAL_FRAMES:
                epoch_frame = 0
                while epoch_frame < FRAMES_BETWEEN_EVAL:
                    start_time = time.time()
                    atari_wrap.reset()
                    life_lost = True
                    episode_reward_sum = 0
                    for _ in range(MAX_EPISODE_LENGTH):
                        action = agent.get_action(frame_number, atari_wrap.state)
                        processed_frame, reward, terminal, life_lost = atari_wrap.step(action)
                        frame_number += 1
                        epoch_frame += 1
                        episode_reward_sum += reward
                        agent.add_experience(action=action,
                                            frame=processed_frame[:, :, 0],
                                            reward=reward, clip_reward=CLIP_REWARD,
                                            terminal=life_lost)
                        if frame_number % UPDATE_FREQUENCY == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                            loss, _ = agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR, frame_number=frame_number, REPLAY_PRIORITY_SCALE=REPLAY_PRIORITY_SCALE)
                            loss_list.append(loss)
                        if frame_number % TARGET_UPDATE_FREQUENCY == 0 and frame_number > MIN_REPLAY_BUFFER_SIZE:
                            agent.update_target_network()
                        if terminal:
                            terminal = 0
                            break
                    rewards.append(episode_reward_sum)
                    if len(rewards) % 10 == 0:
                        tf.summary.scalar('Reward', np.mean(rewards[-10:]), frame_number)
                        tf.summary.scalar('Loss', np.mean(loss_list[-100:]), frame_number)
                        writer.flush()
                        print(f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  Average reward: {np.mean(rewards[-10:]):0.1f}  Time taken: {(time.time() - start_time):.1f}s')
                terminal = True
                eval_rewards = []
                evaluate_frame_number = 0
                for _ in range(EVAL_LENGTH):
                    if terminal:
                        atari_wrap.reset(evaluation=True)
                        life_lost = True
                        episode_reward_sum = 0
                        terminal = False
                    action = 1 if life_lost else agent.get_action(frame_number, atari_wrap.state, evaluation=True) #Breakout Specific Instruction
                    _, reward, terminal, life_lost = atari_wrap.step(action)
                    evaluate_frame_number += 1
                    episode_reward_sum += reward
                    if terminal:
                        eval_rewards.append(episode_reward_sum)
                if len(eval_rewards) > 0:
                    final_score = np.mean(eval_rewards)
                else:
                    final_score = episode_reward_sum
                print('Evaluation score:', final_score)
                tf.summary.scalar('Evaluation score', final_score, frame_number)
                writer.flush()
                if len(rewards) > 300 and SAVE_PATH is not None:
                    agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
    except KeyboardInterrupt:
        writer.close()
        if SAVE_PATH is not None:
            agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)