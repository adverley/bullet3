import argparse
import json
import random
import time
import numpy as np
import os
import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop

from tqdm import tqdm

class ModelAgent:
    def __init__(self, data, lr, n1, n2, bs, target_net_update_freq, zone_queue=False):
        self.lr = lr
        self.state_space = 10
        self.action_space = 4
        self.model = self.create_model(n1, n2)
        self.target_model = self.create_model(n1, n2)

        self.target_net_update_freq = target_net_update_freq
        self.zone_queue = zone_queue

        if zone_queue:
            #self.zone_rewards = {'outside': -10, 'cone': 100, 'torus_col': -10000, 'completion': 10000}
            self.zone_rewards = {'outside': -0.1, 'cone': 1, 'torus_col': -1, 'completion': 10}
            self.zone_dist = [0.25, 0.25, 0.25, 0.25]
            self.zone_idx = [0, 0, 0, 0]

            self.memory_outside = []
            self.memory_cone = []
            self.memory_coll = []
            self.memory_completion = []
            for i in data:
                if int(i[2]) == self.zone_rewards['outside']:
                    self.memory_outside.append(i)
                elif int(i[2]) == self.zone_rewards['cone']:
                    self.memory_cone.append(i)
                elif int(i[2]) == self.zone_rewards['torus_col']:
                    self.memory_coll.append(i)
                elif int(i[2]) == self.zone_rewards['completion']:
                    self.memory_completion.append(i)
                else:
                    print("Unrecognized sample!")
                    exit(-1)
        else:
            self.memory = random.sample(data, len(data))  # deque(maxlen=self.mem_init_size)
        # Create train test set split, memory is train set loss calculated on test set.

        self.batch_idx = 0
        self.batch_size = bs
        self.tau = .125
        self.gamma = 0.999

        #Stats
        self.loss = []
        self.mae = []

    def create_model(self, n1=32, n2=16):
        model = Sequential()
        model.add(Dense(n1, input_dim=self.state_space, activation="relu"))
        model.add(Dense(n2, activation="relu"))
        model.add(Dense(self.action_space))
        print(model.summary())
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.lr), metrics=['mae'])
        return model

    def train_batch(self):
        samples = {
            'state': [],
            'action': [],
            'reward': [],
            'new_state': [],
            'done': []
        }

        _states = []
        _targets = []

        #Since data is shuffled, batch could be subsequent samples
        #sample = random.sample(self.memory, self.batch_size)
        def create_sample(memory, bs, use_idx=None):
            sample = []
            if use_idx is None:
                idx = self.batch_size
            else:
                idx = self.zone_idx[use_idx]

            for i in range(bs):
                sample.append(memory[idx % len(memory)])
                if use_idx is None:
                    self.batch_idx += 1
                    idx = self.batch_size
                else:
                    self.zone_idx[use_idx] += 1
                    idx = self.zone_idx[use_idx]
            return sample

        sample = []

        if self.zone_queue:
            bs = self.batch_size
            sample1 = create_sample(self.memory_outside, int(self.batch_size*self.zone_dist[0]), 0)
                #random.sample(self.memory_outside, int(self.batch_size*self.zone_dist[0]))
            bs -= int(self.batch_size*self.zone_dist[0])
            sample2 = create_sample(self.memory_cone, int(self.batch_size*self.zone_dist[1]), 1)
                #random.sample(self.memory_cone, int(self.batch_size*self.zone_dist[1]))
            bs -= int(self.batch_size * self.zone_dist[1])
            sample3 = create_sample(self.memory_coll, int(self.batch_size*self.zone_dist[2]), 2)
                #random.sample(self.memory_coll, int(self.batch_size*self.zone_dist[2]))
            bs -= int(self.batch_size * self.zone_dist[2])
            sample4 = create_sample(self.memory_completion, bs, 3)
                #random.sample(self.memory_completion, bs)

            #print("\nBatch composition:", len(sample1), len(sample2), len(sample3), len(sample4))
            sample = sample1 + sample2 + sample3 + sample4
        else:
            sample = create_sample(self.memory, self.batch_size)
            #for i in range(self.batch_size):
            #    sample.append(self.memory[self.batch_idx % len(self.memory)])
            #    self.batch_idx += 1

        for s in sample:
            #state, action, reward, new_state, done = sample
            samples['state'].append(s[0][0])
            samples['action'].append(s[1])
            samples['reward'].append(s[2])
            samples['new_state'].append(s[3][0])
            samples['done'].append(s[4])

        targets = self.model.predict_on_batch(np.array(samples['state']))
        q_vals = self.model.predict_on_batch(np.array(samples['new_state']))

        for_set = zip(targets, q_vals, samples['state'], samples['new_state'],
                      samples['action'], samples['reward'], samples['done'])

        for target, q_val, state, new_state, action, reward, done in for_set:
            target[action] = reward

            _states.append(state)
            _targets.append(target)

        # Take only the last value of each episode
        losses = self.model.train_on_batch(np.array(_states), np.array(_targets))
        self.loss.append(float(losses[0]))
        self.mae.append(float(losses[1]))

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def print_stats(self, ep, ep_tot, time, steps):
        if steps == 0:
            steps += 1

        print("\n{}/{}".format(ep, ep_tot),
              "Execution time: {} steps/s".format(round(steps/time, 2)),
              "Mae:", np.average(self.mae),
              "Loss:", np.average(self.loss),
              "\n"
             ) #trail_len

def main(args):
    configs = [
                #Test impact update_freq
                #(0.0001, 32, 16, 32, 10000),
                (0.0001, 32, 16, 64, 10000),
                #(0.0001, 32, 16, 128, 10000),
                #(0.0001, 64, 32, 32, 10000),
                (0.0001, 64, 32, 64, 10000),
                #(0.0001, 64, 32, 128, 10000),

                #Test impact larger network
                #(0.0001, 1024, 512, 32, 10000),
                (0.0001, 1024, 512, 64, 10000),
                #(0.0001, 1024, 512, 128, 10000),

               ] #(lr, n1, n2, bs, gamma)

    results = [] #(exp, los, mae)

    EPISODES = 10000  # env.nb_episodes
    trial_len = 400  # env.nb_steps

    filepath_experiment = "../../baselines/experiments/"
    exp_counter = 0
    metrics = {
                'loss': [],
                'mae': []
            }

    with open(os.path.join(filepath_experiment, args.f)) as fp:
        data = json.load(fp)

    for exp in tqdm(configs):
        agent = ModelAgent(data, *exp, zone_queue=args.zone_queue)
        tot_step = 0

        for ep in tqdm(range(EPISODES)):
            start_time = time.time()
            for step in range(trial_len):
                agent.train_batch() #Internally iterates default (prediction) model
                tot_step += 1  # Used to determine replay memory update

            if ep % 2000 == 0:
                agent.print_stats(ep, EPISODES, time.time() - start_time, step)
            metrics['loss'].append(np.average(agent.loss))
            metrics['mae'].append(np.average(agent.mae))
            agent.loss = []
            agent.mae = []

        results.append((exp_counter, metrics['loss'], metrics['mae']))
        exp_counter += 1
        metrics = {
            'loss': [],
            'mae': []
        }

        #Write results to file
        print("Saving data...")
        out_fn = os.path.join(filepath_experiment, args.out)
        save_output(out_fn, results)

def save_output(fn, output):
    def default(o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError

    with open(fn, 'w') as fp:
        json.dump(output, fp, default=default)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='baxter_dqn_exp26_config_net2layer_small_mem_log.json')
    parser.add_argument('-out', type=str, default='test')
    parser.add_argument('-zone_queue', type=bool, default=False)
    args = parser.parse_args()
    main(args)