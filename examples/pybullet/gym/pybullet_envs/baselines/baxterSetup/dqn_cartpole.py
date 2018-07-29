# -*- coding: utf-8 -*-
import random
import gym
import json
import sys
import time
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

        self.metrics = {
            'episode': [],
            'episode_reward': [],
            'min_reward': [],
            'max_reward': [],
            'mean_q': [],
            'max_q': [],
            'min_q': [],
        }

        self.cs_reward = 0
        self.bound_reward = [sys.maxsize, -sys.maxsize - 1]
        self.cs_qval = 0
        self.bound_qval = [0, 0]

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # Reward stats
        self.bound_reward = [min(reward, self.bound_reward[0]),
                             max(reward, self.bound_reward[1])]
        self.cs_reward += reward
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        #Stats
        mean_qval = 0
        bound_qval = [0, 0]

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            q_vals = self.model.predict(next_state)[0]
            if not done:
                #print("Q-vals:", q_vals)
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            mean_qval += np.mean(q_vals)
            bound_qval[0] += min(q_vals)
            bound_qval[1] += max(q_vals)

            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Stats
        self.cs_qval += mean_qval / float(batch_size)
        self.bound_qval[0] += bound_qval[0] / float(batch_size)
        self.bound_qval[1] += bound_qval[1] / float(batch_size)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def save_data(self, fn):
        def default(o):
            if isinstance(o, np.int64): return int(o)
            raise TypeError

        with open(fn, 'w') as fp:
            json.dump(self.metrics, fp, default=default)

    def print_stats(self, ep, ep_tot, trial_len, time, steps):
        if steps == 0:
            steps += 1
        mean_q = float(round(self.cs_qval / steps, 4))
        mean_bound_q = [float(round(x / steps, 4)) for x in self.bound_qval]
        mean_bound_reward = [round(x, 4) for x in self.bound_reward]

        print("\n{}/{}".format(ep, ep_tot),
              "Execution time: {} steps/s".format(round(steps/time, 2)),
              "Episode reward:", self.cs_reward, mean_bound_reward,
              "Mean Q:", mean_q, mean_bound_q,
              "Curr epsilon:", self.epsilon,
             ) #trail_len

        self.metrics['episode'].append(ep)
        self.metrics['episode_reward'].append(self.cs_reward)
        self.metrics['min_reward'].append(self.bound_reward[0])
        self.metrics['max_reward'].append(self.bound_reward[1])
        self.metrics['mean_q'].append(mean_q)
        self.metrics['min_q'].append(mean_bound_q[0])
        self.metrics['max_q'].append(mean_bound_q[1])

        # Reset Statistics
        self.cs_reward = 0
        self.bound_reward = [sys.maxsize, -sys.maxsize - 1]
        self.cs_qval = 0
        self.bound_qval = [0, 0]

if __name__ == "__main__":
    filepath_experiment = "../../baselines/experiments/"
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("State and action space:", state_size, action_size)
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        start_time = time.time()
        for t in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, t, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

        agent.print_stats(e, EPISODES, 500, time.time() - start_time, t)
        if e % 500:
            agent.save_data(os.path.join(filepath_experiment, 'dqn_{}_data.json'.format("cartpole")))

    agent.save_data(os.path.join(filepath_experiment, 'dqn_{}_data.json'.format("cartpole")))