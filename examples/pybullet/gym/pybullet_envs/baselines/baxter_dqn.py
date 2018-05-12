import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
import argparse

from pybullet_envs.bullet.baxterGymEnv import BaxterGymEnv
from pybullet_envs.bullet.callbacks import DataLogger

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.processors import WhiteningNormalizerProcessor

import datetime
import numpy as np
import time
import string

def int2base(x, base):
    digs = string.digits + string.ascii_letters
    digits = []

    if x < 0:
        raise ValueError('Got negative number: {}'.format(str(x)))
    elif x == 0:
        return digs[0]

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    digits.reverse()
    return ''.join(digits)

def main(args):
    ENV_NAME = "BaxterGymEnv"
    EXP_NAME = args.exp_name
    WINDOW_LENGTH = 1

    filepath_experiment = "experiments/"

    env = BaxterGymEnv(
            renders=False,
            isDiscrete=True,
            useCamera=False,
            maxSteps=400
            )
    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    print("Action size:", action_size)
    print("State size:", state_size)

    assert len(env.action_space.shape) == 1
    # Interpret result as base 3 number to index arrays
    nb_actions = int('2222222', 3)

    model = Sequential()
    model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = WhiteningNormalizerProcessor()

    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)

    # The trade-off between exploration and exploitation is difficult and an on-going research topic.
    # If you want, you can experiment with the parameters or use a different policy. Another popular one
    # is Boltzmann-style exploration:
    # policy = BoltzmannQPolicy(tau=1.)

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    if args.mode == 'train':
        weights_filename = 'dqn_{}_{}_weights.h5f'.format(ENV_NAME, EXP_NAME)
        checkpoint_weights_filename = 'dqn_{}_{}_weights_checkpoint.h5f'.format(ENV_NAME, EXP_NAME)
        log_filename = 'dqn_{}_{}_log.json'.format(ENV_NAME, EXP_NAME)
        data_filename = filepath_experiment + 'ddpg_{}_{}_data.json'.format(ENV_NAME, EXP_NAME)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        callbacks += [DataLogger(data_filename)]
        dqn.fit(env, callbacks=callbacks, nb_steps=2000000, log_interval=10000)

        # After training is done, we save the final weights one more time.
        dqn.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        dqn.test(env, nb_episodes=10, visualize=False)

    elif args.mode == 'test':
        weights_filename = 'dqn_{}_{}_weights.h5f'.format(ENV_NAME, EXP_NAME)
        dqn.load_weights(weights_filename)
        dqn.test(env, nb_episodes=10, visualize=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--exp-name', type=str, required=True)
    args = parser.parse_args()
    main(args)
