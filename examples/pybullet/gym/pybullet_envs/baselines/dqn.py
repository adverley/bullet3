import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import argparse
import gym
import numpy as np
import random
import sys
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from pybullet_envs.bullet.baxterGymEnv import BaxterGymEnv
from collections import deque

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.05
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

        #Statistics
        self.cs_action = 0
        self.bound_action = [sys.maxsize, -sys.maxsize - 1]
        self.cs_reward = 0
        self.bound_reward = [sys.maxsize, -sys.maxsize - 1]
        self.cs_qval = 0
        self.bound_qval = [sys.maxsize, -sys.maxsize - 1]

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        print(model.summary())
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.model.predict(state)[0])

        self.cs_action += action
        self.bound_action = [min(self.bound_action[0], action),
                             max(self.bound_action[1], action)]
        return action

    def remember(self, state, action, reward, new_state, done):
        self.bound_reward = [min(reward, self.bound_reward[0]),
                             max(reward, self.bound_reward[1])]
        self.cs_reward += reward
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = {
            'state': [],
            'action': [],
            'reward': [],
            'new_state': [],
            'done': []
        }

        _states = []
        _targets = []

        mean_qval = 0
        bound_qval = [0, 0]

        sample = random.sample(self.memory, batch_size)
        for s in sample:
            #state, action, reward, new_state, done = sample
            samples['state'].append(s[0][0])
            samples['action'].append(s[1])
            samples['reward'].append(s[2])
            samples['new_state'].append(s[3][0])
            samples['done'].append(s[4])

        # Update is done to late so what's the point in using the target network principle here?
        targets = self.target_model.predict_on_batch(np.array(samples['state']))
        q_vals = self.target_model.predict_on_batch(np.array(samples['new_state']))

        for target, q_val, state, new_state, action, reward, done in zip(targets, q_vals, samples['state'], samples['new_state'],
                                                                            samples['action'], samples['reward'], samples['done']):
            mean_qval += np.mean(target)
            bound_qval[0] += min(target)
            bound_qval[1] += max(target)

            if done:
                target[action] = reward
            else:
                Q_future = max(q_val)
                target[action] = reward + Q_future * self.gamma

            _states.append(state)
            _targets.append(target)

        self.model.train_on_batch(np.array(_states), np.array(_targets))

        # Stats
        self.cs_qval += mean_qval / float(batch_size)
        self.bound_qval[0] += bound_qval[0] / float(batch_size)
        self.bound_qval[1] += bound_qval[1] / float(batch_size)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

    def load_model(self, name):
        self.model.load_weights(name)

    def print_stats(self, ep, ep_tot, trial_len, time, steps):
        print(" {}/{}".format(ep, ep_tot),
              "Execution time: {} steps/s".format(round(steps/time, 2)),
              "Episode reward:", self.cs_reward, self.bound_reward,
              "Mean action:", self.cs_action / trial_len, self.bound_action,
              "Mean Q value:", round(self.cs_qval / trial_len, 4), self.bound_qval
             ) #trail_len

        # Reset Statistics
        self.cs_action = 0
        self.bound_action = [sys.maxsize, -sys.maxsize - 1]
        self.cs_reward = 0
        self.bound_reward = [sys.maxsize, -sys.maxsize - 1]
        self.cs_qval = 0
        self.bound_qval = [sys.maxsize, -sys.maxsize - 1]


def main(args):
    EXP_NAME = args.exp_name
    REWARD = args.reward
    WINDOW_LENGTH = 1

    #env = gym.make("MountainCar-v0")
    env = BaxterGymEnv(
            renders=args.render,
            useCamera=False,
            maxSteps=400,
            dv=0.2,
            _algorithm='DQN',
            _reward_function=REWARD,
            _action_type='single'
            )

    EPISODES  = 2000 #env.nb_episodes
    trial_len = env._maxSteps  #env.nb_steps

    filepath_experiment = "experiments/"

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("State size", state_size)
    print("Action size", action_size)

    dqn_agent = DQNAgent(env=env)
    # fn = os.path.join(filepath_experiment, 'baxter_dqn_checkpoint_dqn_test.h5f')
    # dqn_agent.load_model(fn)

    for ep in range(EPISODES):
        cur_state = env.reset().reshape(1, state_size)
        start_time = time.time()
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            # reward = reward if not done else -20
            new_state = new_state.reshape(1,state_size)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()       # internally iterates default (prediction) model
            dqn_agent.target_train() # iterates target model

            cur_state = new_state
            if done:
                break

        dqn_agent.print_stats(ep, EPISODES, trial_len, time.time()-start_time, step)
        if step < trial_len-1:
            print("Completed in {} steps".format(step))

        if ep % 100 == 0:
            weights_filename = os.path.join(filepath_experiment, "baxter_dqn_checkpoint_{}.h5f".format(EXP_NAME))
            dqn_agent.save_model(weights_filename)

    weights_filename = os.path.join(filepath_experiment, "baxter_dqn_{}.h5f".format(EXP_NAME))
    dqn_agent.save_model(weights_filename)

if __name__ == "__main__":
    reward_functions = [
                        'finite_line_distance',
                        'line_distance',
                        'sparse',
                        'torus_distance'
                        ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--exp_name', type=str, default="test")#required=True)
    parser.add_argument('--reward', type=str, choices=reward_functions, default=reward_functions[0])
    args = parser.parse_args()
    main(args)
