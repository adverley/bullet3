# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from pybullet_envs.bullet.baxterGymEnv import BaxterGymEnv

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Permute
from keras.layers.convolutional import Convolution2D, Conv2D, MaxPooling2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from numpy.random import normal

import datetime
import numpy as np


def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    # print("totalt")
    # print(totalt)
    is_solved = totalt > 2000 and total >= 10
    return is_solved


def build_model():
    """ # Model from the DeepMind paper --> look for subsampling
    model = Sequential()
    # (samples, height, width, channels)
    # model.add(Permute((1, 3, 2), input_shape=input_shape))
    model.add(Conv2D(64, 7, 2, activation='relu',
                     input_shape=env.observation_space.shape))
    model.add(Conv2D(32, 5, 1, activation='relu'))
    model.add(Conv2D(32, 5, 1, activation='relu'))
    model.add(Activation('softmax'))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(40))
    model.add(Dense(40))
    # model.add(Dense(nb_actions, activation='linear'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    return model
    """


def main():
    env = BaxterGymEnv(renders=False, isDiscrete=True)
    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    print "Action size:", action_size
    print "State size:", state_size
    episodes = 1000
    batch_size = 32
    WINDOW_LENGTH = 4
    ENV_NAME = "BaxterGymEnv"

    nb_actions = env.action_space.shape[1]
    print "Action space:", nb_actions
    print "Observation space:", env.observation_space.shape
    # (height, width, channel)
    # input_shape = (1,) + env.observation_space.shape

    INPUT_SHAPE = state_size

    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = Sequential()
    model.add(Permute((1, 3, 2), input_shape=input_shape))
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary()

    memory=SequentialMemory(limit=1000000, window_length=1)
    policy=LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)
    dqn=DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, nb_steps_warmup=50000,
                   gamma=.99, target_model_update=10000, train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    dqn.fit(env, nb_steps=1750000, visualize=True, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    main()
