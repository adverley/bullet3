# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from pybullet_envs.bullet.baxterGymEnv import BaxterGymEnv
from pybullet_envs.bullet.callbacks import DataLogger

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Permute, Input, Concatenate
from keras.layers.convolutional import Convolution2D, Conv2D, MaxPooling2D
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.random import OrnsteinUhlenbeckProcess
from rl.processors import WhiteningNormalizerProcessor

import datetime
import numpy as np
import time

# save experiment vars
timestr = time.strftime("%Y%m%d-%H%M%S")
#filepath_experiment = "/experiments/ddpg_{}/".format(timestr)
filepath_experiment = "experiments/"


baxterProcessor = WhiteningNormalizerProcessor()


def main():
    # Train network with joint position inputs
    env = BaxterGymEnv(renders=False, isDiscrete=True,
                       useCamera=False, maxSteps=400)
    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    print "Action size:", action_size
    print "State size:", state_size
    episodes = 1000
    batch_size = 32
    WINDOW_LENGTH = 1
    ENV_NAME = "BaxterGymEnv"

    print "Action space:", env.action_space.shape
    print "Observation space:", env.observation_space.shape
    # (height, width, channel)
    # input_shape = (1,) + env.observation_space.shape

    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
    actor.add(Dense(1024))
    actor.add(Activation('relu'))
    actor.add(Dense(512))
    actor.add(Activation('relu'))
    actor.add(Dense(256))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('tanh'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(
        shape=(WINDOW_LENGTH,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Dense(1024)(flattened_observation)
    x = Activation('relu')(x)
    x = Concatenate()([x, action_input])
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    random_process = OrnsteinUhlenbeckProcess(
        size=nb_actions, theta=.15, mu=0., sigma=.1)

    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                      random_process=random_process, gamma=.99, target_model_update=1e-3,
                      processor=baxterProcessor)
    agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])

    # Initialize callbacks
    callbacks = []
    log_filename = filepath_experiment + 'ddpg_{}_log.json'.format(ENV_NAME)
    callbacks += [FileLogger(log_filename, interval=1)]

    # log all train data with custom callback
    data_filename = filepath_experiment + 'ddpg_{}_data.json'.format(ENV_NAME)
    callbacks += [DataLogger(data_filename)]

    # make model checkpoints
    checkpoint_filename = os.path.join(
        filepath_experiment, 'ddpg_{}_weights_checkpoint.h5f'.format(ENV_NAME))
    callbacks += [ModelIntervalCheckpoint(checkpoint_filename,
                                          interval=10000, verbose=1)]

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    agent.fit(env, nb_steps=1000000, visualize=False,
              callbacks=callbacks, verbose=2)

    # After training is done, we save the final weights.
    weights_filename = os.path.join(
        filepath_experiment, 'ddpg_{}_weights.h5f'.format(ENV_NAME))
    agent.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)


if __name__ == '__main__':
    main()
