************************README************************
This program trains a Dueling DQN on Atari Games. Currently it supports Breakout. 
Later we can easily add support for other games too. To run it, first install these
dependencies:
1. Tensorflow 2.3.0 (GPU)
2. OpenCV
3. OpenAI Gym
4. Gym support for Atari

After installing these you can just 
1. For training the model run python3 train_dqn.py. In case you are trying to train from a previous checkpoint
add a cmdline argument for its location, e.g python3 train_dqn.py saves/save-000/.
2. For evaluating a trained model just run 
python3 evaluation.py location_of_model.

The files in the folder contain code for:
1. agent_dueling_dqn.py: Class for Dueling DQN
2. AtariWrapper: Wrapper Class for Gym[Atari].
3. Config.py: File for global constants. You can modify those variables. Some variables are hyperparameters for
further tuning.
4. Evaluation.py: Code for testing your model in a game visually!.
5. ReplayBuffer.py: Class for replayBuffer. In future will support prioritized experience replay.
6. train_dqn.py: Main file for training the model.