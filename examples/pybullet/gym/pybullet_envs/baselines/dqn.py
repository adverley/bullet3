import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import argparse
import json
import numpy as np
import random
import sys
import time
import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop

from tqdm import tqdm

from dqn_config import DQNConfig
from pybullet_envs.bullet.baxterGymEnv import BaxterGymEnv
from collections import deque

config = DQNConfig()
experiments = config.experiments

class DQNAgent:
    def __init__(self, env, exp_num):
        exp = experiments[exp_num]
        print(exp)

        self.env = env
        self.memory = deque(maxlen=exp['memory_size'])

        self.gamma = exp['gamma'] #0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = exp['epsilon_decay'] #0.995
        self.epsilon_guided = exp['epsilon_guided']
        self.learning_rate = exp['learning_rate'] #0.05
        self.tau = .125

        self.replay_mem_init_size = exp['replay_mem_init_size']
        self.replay_mem_update_freq = exp['replay_mem_update_freq']

        self.loss_optimizer = exp['loss_function']

        self.model = self.create_model()
        self.target_model = self.create_model()

        #Statistics
        self.cs_action = 0
        self.bound_action = [sys.maxsize, -sys.maxsize - 1]
        self.cs_reward = 0
        self.bound_reward = [sys.maxsize, -sys.maxsize - 1]
        self.cs_qval = 0
        self.bound_qval = [0, 0]
        self.mae = -1.
        self.loss = -1.

        self.metrics = {
            'episode': [],
            'episode_reward': [],
            'min_reward': [],
            'max_reward': [],
            'mean_q': [],
            'max_q': [],
            'min_q': [],
            'mean_action': [],
            'max_action': [],
            'min_action': [],
            'epsilon': [],
            'mae': [],
            'loss': []
        }

    def create_model(self):
        def huber_loss(y_true, y_pred, clip_delta=1.0):
            error = y_true - y_pred
            cond = tf.keras.backend.abs(error) < clip_delta

            squared_loss = 0.5 * tf.keras.backend.square(error)
            linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

            return tf.where(cond, squared_loss, linear_loss)

        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(512, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        print(model.summary())
        if self.loss_optimizer == 'mse':
            model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate), metrics=['mae'])
        elif self.loss_optimizer == 'huber':
            model.compile(loss=huber_loss, optimizer=RMSprop(lr=self.learning_rate), metrics=['mae'])
        else:
            raise NotImplementedError
        return model

    def update_exploration(self): #Epsilon was decaying to fast when it was in act, is this better?
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def act(self, state):
        if np.random.random() < self.epsilon:
            if self.epsilon_guided:
                action = self.env.getGuidedAction()
            else:
                action = self.env.action_space.sample()

        else:
            action = np.argmax(self.model.predict(state)[0])

        self.cs_action += action
        self.bound_action = [min(self.bound_action[0], action),
                             max(self.bound_action[1], action)]
        return action

    def test(self, state):
        q_values = self.model.predict(state)[0]
        return np.argmax(q_values)

    def remember(self, state, action, reward, new_state, done):
        self.bound_reward = [min(reward, self.bound_reward[0]),
                             max(reward, self.bound_reward[1])]
        self.cs_reward += reward
        self.memory.append([state, action, reward, new_state, done])

    def init_replay_mem(self):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        cur_state = self.env.reset().reshape(1, state_size)

        print("Starting replay memory initialization")
        for i in tqdm(range(self.replay_mem_init_size)):
            action = np.random.randint(0, action_size)
            new_state, reward, done, _ = self.env.step(action)

            new_state = new_state.reshape(1, state_size)
            self.memory.append([cur_state, action, reward, new_state, done])
            cur_state = new_state

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

        # Take only the last value of each episode
        losses = self.model.train_on_batch(np.array(_states), np.array(_targets))
        self.loss = losses[0]
        self.mae = losses[1]

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

    def save_data(self, fn):
        def default(o):
            if isinstance(o, np.int64): return int(o)
            raise TypeError

        with open(fn, 'w') as fp:
            json.dump(self.metrics, fp, default=default)

    def load_data(self, fn):
        with open(fn, 'r') as fp:
            data = json.load(fp)
        return data

    def save_model(self, fn):
        #self.model.save(fn)
        self.model.save_weights(fn, overwrite=True)

    def load_model(self, name):
        self.model.load_weights(name)
        self.target_model.set_weights(self.model.get_weights())

    def print_stats(self, ep, ep_tot, trial_len, time, steps):
        mean_q = float(round(self.cs_qval / trial_len, 4))
        mean_bound_q = [float(round(x / trial_len, 4)) for x in self.bound_qval]
        mean_action = self.cs_action / trial_len
        mean_bound_reward = [round(x, 4) for x in self.bound_reward]

        print("\n{}/{}".format(ep, ep_tot),
              "Execution time: {} steps/s".format(round(steps/time, 2)),
              "Episode reward:", self.cs_reward, mean_bound_reward,
              "Mean action:", mean_action, self.bound_action,
              "Mean Q:", mean_q, mean_bound_q,
              "Curr epsilon:", self.epsilon,
              "Mae:", self.mae,
              "Loss:", self.loss
             ) #trail_len

        self.metrics['episode'].append(ep)
        self.metrics['episode_reward'].append(self.cs_reward)
        self.metrics['min_reward'].append(self.bound_reward[0])
        self.metrics['max_reward'].append(self.bound_reward[1])
        self.metrics['mean_q'].append(mean_q)
        self.metrics['min_q'].append(mean_bound_q[0])
        self.metrics['max_q'].append(mean_bound_q[1])
        self.metrics['mean_action'].append(mean_action)
        self.metrics['min_action'].append(self.bound_action[0])
        self.metrics['max_action'].append(self.bound_action[1])
        self.metrics['epsilon'].append(self.epsilon)
        self.metrics['mae'].append(self.mae)
        self.metrics['loss'].append(self.loss)

        # Reset Statistics
        self.cs_action = 0
        self.bound_action = [sys.maxsize, -sys.maxsize - 1]
        self.cs_reward = 0
        self.bound_reward = [sys.maxsize, -sys.maxsize - 1]
        self.cs_qval = 0
        self.bound_qval = [0, 0]
        self.mae = -1
        self.loss = -1


def main(args):
    EXP_NAME = args.exp_name
    REWARD = args.reward
    EXP_NUM = args.exp_num
    WINDOW_LENGTH = 1

    EXP = experiments[EXP_NUM]

    #env = gym.make("MountainCar-v0")
    env = BaxterGymEnv(
            renders=args.render,
            useCamera=False,
            useRandomPos=EXP['randomPos'],
            maxSteps=400,
            dv=0.1,
            _algorithm='DQN',
            _reward_function=EXP['reward'],
            _action_type='single'
            )

    EPISODES  = 10000 #env.nb_episodes
    trial_len = env._maxSteps  #env.nb_steps

    filepath_experiment = "experiments/"

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("State size", state_size)
    print("Action size", action_size)

    dqn_agent = DQNAgent(env=env, exp_num=EXP_NUM)
    # fn = os.path.join(filepath_experiment, 'baxter_dqn_checkpoint_dqn_test.h5f')
    # dqn_agent.load_model(fn)

    if args.mode == 'train':
        try:
            dqn_agent.init_replay_mem()
        except KeyboardInterrupt:
            print("Interrupted initialization, continuing with training...")

        #load pre-trained weights
        if args.pre_name is not None:
            fn = os.path.join(filepath_experiment, "baxter_dqn_{}.h5f".format(args.pre_name))
            dqn_agent.load_model(fn)

        tot_step = 0

        for ep in range(EPISODES):
            cur_state = env.reset().reshape(1, state_size)
            start_time = time.time()
            for step in range(trial_len):
                action = dqn_agent.act(cur_state)
                new_state, reward, done, _ = env.step(action)

                new_state = new_state.reshape(1,state_size)
                dqn_agent.remember(cur_state, action, reward, new_state, done)

                dqn_agent.replay()       # internally iterates default (prediction) model
                tot_step += 1            # Used to determine replay memory update

                if tot_step % dqn_agent.replay_mem_update_freq == 0:
                    print("Updating target network...")
                    dqn_agent.target_train() # iterates target model

                cur_state = new_state

                if done:
                    break

            if ep > EXP['epsilon_start']:
                dqn_agent.update_exploration()

            if ep % 2000 == 0:
                print("Saving data...")
                dqn_agent.save_data(os.path.join(filepath_experiment, 'baxter_dqn_{}_data.json'.format(EXP_NAME)))

            dqn_agent.print_stats(ep, EPISODES, trial_len, time.time()-start_time, step)
            if step < trial_len-1:
                print("Completed in {} steps".format(step))

            if ep % 100 == 0:
                weights_filename = os.path.join(filepath_experiment, "baxter_dqn_checkpoint_{}.h5f".format(EXP_NAME))
                dqn_agent.save_model(weights_filename)

        print("Saving weights and data...")
        weights_filename = os.path.join(filepath_experiment, "baxter_dqn_{}.h5f".format(EXP_NAME))
        dqn_agent.save_model(weights_filename)
        dqn_agent.save_data(os.path.join(filepath_experiment, 'baxter_dqn_{}_data.json'.format(EXP_NAME)))

    elif args.mode == 'test':
        # load weights
        fn = os.path.join(filepath_experiment, "baxter_dqn_{}.h5f".format(EXP_NAME))
        dqn_agent.load_model(fn)

        EPISODES = 10
        trial_len = env._maxSteps

        for ep in range(EPISODES):
            cur_state = env.reset().reshape(1, state_size)
            start_time = time.time()
            ep_reward = 0
            bound_reward = [sys.maxsize, -sys.maxsize - 1]
            ep_action = 0
            bound_action = [sys.maxsize, -sys.maxsize - 1]

            for step in range(trial_len):
                action = dqn_agent.test(cur_state)
                print("Next action:", np.argmax(dqn_agent.target_model.predict(cur_state)[0]))
                new_state, reward, done, _ = env.step(action)

                # Stats
                ep_reward += reward
                ep_action += action
                bound_reward = [min(bound_reward[0], reward), max(bound_reward[1], reward)]
                bound_action = [min(bound_action[0], action), max(bound_action[1], action)]

                new_state = new_state.reshape(1, state_size)
                cur_state = new_state

                if done:
                    break

            #dqn_agent.print_stats(ep, EPISODES, trial_len, time.time() - start_time, step)
            print("\n{}/{}".format(ep, EPISODES),
                  "Execution time: {} steps/s".format(round(trial_len / (time.time() - start_time), 2)),
                  "Episode reward:", ep_reward, [round(x, 4) for x in bound_reward],
                  "Mean action:", ep_action / trial_len, bound_action,
                  )

            if step < trial_len - 1:
                print("Completed in {} steps".format(step))

if __name__ == "__main__":
    reward_functions = [
                        'finite_line_distance',
                        'line_distance',
                        'sparse',
                        'torus_distance',
                        'clipped_reward'
                        ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--exp_name', type=str, default="test")#required=True)
    parser.add_argument('--reward', type=str, choices=reward_functions, default=reward_functions[0])
    parser.add_argument('--exp_num', type=int, default=0)
    parser.add_argument('--pre_name', type=str, default=None)
    args = parser.parse_args()
    main(args)
