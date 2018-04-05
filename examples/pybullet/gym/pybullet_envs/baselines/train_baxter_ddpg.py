# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from pybullet_envs.bullet.baxterGymEnv import BaxterGymEnv

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

from PIL import Image

import datetime
import numpy as np
import time

INPUT_SHAPE = (240, 240)

# save experiment vars
timestr = time.strftime("%Y%m%d-%H%M%S")
#filepath_experiment = "/experiments/ddpg_{}/".format(timestr)
filepath_experiment = "experiments/"


class BaxterProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation.astype('uint8'))
        img = img.resize(INPUT_SHAPE).convert(
            'L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        # saves storage in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch


def main():
    # Train network with joint position inputs
    env = BaxterGymEnv(renders=True, isDiscrete=True,
                       useCamera=True, maxSteps=200)
    ENV_NAME = "BaxterGymEnv"

    WINDOW_LENGTH = 1
    INPUT_SHAPE = (240, 240)
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

    print "Action space:", env.action_space.shape
    print "Observation space:", env.observation_space.shape
    # (height, width, channel)
    # input_shape = (1,) + env.observation_space.shape

    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Permute((1, 3, 2), input_shape=input_shape))
    actor.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
    actor.add(Activation('relu'))
    actor.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    actor.add(Activation('relu'))
    actor.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    actor.add(Activation('relu'))
    actor.add(Flatten())
    actor.add(Dense(512))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(model.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(
        shape=(WINDOW_LENGTH,) + INPUT_SHAPE, name='observation_input')

    x = Conv2D(64, 7, 2, activation='relu')(observation_input)
    x = Conv2D(32, 5, 1, activation='relu')(x)
    x = Conv2D(32, 5, 1, activation='softmax')(x)
    flattened_observation = Flatten()(x)
    x = Dense(64)(flattened_observation)
    x = Activation('relu')(x)
    x = Concatenate()([x, action_input])
    x = Dense(40)(x)
    x = Activation('relu')(x)
    x = Dense(40)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    random_process = OrnsteinUhlenbeckProcess(
        size=nb_actions, theta=.15, mu=0., sigma=.1)

    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                      random_process=random_process, gamma=.99, target_model_update=1e-3,
                      processor=BaxterProcessor())
    agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])

    # Initialize callbacks
    callbacks = []
    log_filename = filepath_experiment + 'ddpg_{}_log.json'.format(ENV_NAME)
    callbacks += [FileLogger(log_filename, interval=1)]

    # log all train data with custom callback
    #callbacks += [DataLogger(filepath_experiment, interval=100)]

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
