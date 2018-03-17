# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Permute
from keras.layers.convolutional import Convolution2D, Conv2D, MaxPooling2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

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


def main():
    env = KukaCamGymEnv(renders=False, isDiscrete=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    episodes = 1000
    batch_size = 32
    WINDOW_LENGTH = 4
    ENV_NAME = "KukaCamGymEnv"

    """
    agent = DQNAgent(state_size, action_size)

    # Iterate the game
    for e in range(episodes):
        # reset state in the beginning of each game
        state = env._reset()
        state = np.reshape(state, [1, state_size])
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            # turn this on if you want to render
            # env.render()
            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env._step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, time_t))
                break
        # train the agent with the experience of the episode
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if e % 10 == 0:
            agent.save("./save/kukagrasp-dqn.h5")
    """

    nb_actions = env.action_space.n
    print "Action space:", nb_actions
    print "Observation space:", env.observation_space.shape
    # (height, width, channel)
    input_shape = (1,) + env.observation_space.shape

    model = Sequential()
    # (samples, height, width, channels)
    #model.add(Permute((1, 3, 2), input_shape=input_shape))
    model.add(Conv2D(64, 7, 2, activation='relu',
                     input_shape=env.observation_space.shape))
    model.add(Conv2D(32, 5, 1, activation='relu'))
    model.add(Conv2D(32, 5, 1, activation='relu'))
    model.add(Activation('softmax'))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(40))
    model.add(Dense(40))
    #model.add(Dense(nb_actions, activation='linear'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    memory = SequentialMemory(limit=1000000, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, nb_steps_warmup=50000,
                   gamma=.99, target_model_update=10000, train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    dqn.fit(env, nb_steps=1750000, visualize=True, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    main()
