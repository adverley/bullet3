import argparse
import json
import random
import time
import numpy as np
import os

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop

from collections import deque
from tqdm import tqdm

#Use similar method as in dqn but for 400 iterations take samples for deque memory and use another
#sample to test and after 400 iterations flush the buffer and take new samples?

#Tell later never flushed completely after each episode --> just appended samples to the buffer after each step
# Not enough variation?

class ModelAgent:
    def __init__(self, data, lr, n1, n2, bs, target_net_update_freq):
        self.lr = lr
        self.state_space = 10
        self.action_space = 4
        self.model = self.create_model(n1, n2)
        self.target_model = self.create_model(n1, n2)

        self.mem_init_size = 200000
        self.target_net_update_freq = target_net_update_freq
        self.memory = deque(maxlen=self.mem_init_size)
        self.batch_size = bs
        self.tau = .125
        self.gamma = 0.999

        self.data = data
        self.cur_idx = 0

        #Stats
        self.loss = -1.
        self.mae = -1.

    def create_model(self, n1=32, n2=16):
        model = Sequential()
        model.add(Dense(n1, input_dim=self.state_space, activation="relu"))
        model.add(Dense(n2, activation="relu"))
        model.add(Dense(self.action_space))
        print(model.summary())
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.lr), metrics=['mae'])
        return model

    def add_sample(self):
        self.memory.append(self.data[self.cur_idx])
        done = self.data[self.cur_idx][4]
        self.cur_idx += 1
        return done

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

        sample = random.sample(self.memory, self.batch_size)
        for s in sample:
            #state, action, reward, new_state, done = sample
            samples['state'].append(s[0][0])
            samples['action'].append(s[1])
            samples['reward'].append(s[2])
            samples['new_state'].append(s[3][0])
            samples['done'].append(s[4])

        targets = self.target_model.predict_on_batch(np.array(samples['state']))
        q_vals = self.target_model.predict_on_batch(np.array(samples['new_state']))

        for_set = zip(targets, q_vals, samples['state'], samples['new_state'],
                      samples['action'], samples['reward'], samples['done'])

        for target, q_val, state, new_state, action, reward, done in for_set:
            if done:
                target[action] = reward
            else:
                Q_future = max(q_val)
                target[action] = reward + Q_future * self.gamma

            _states.append(state)
            _targets.append(target)

        # Take only the last value of each episode
        losses = self.model.train_on_batch(np.array(_states), np.array(_targets))
        self.loss = float(losses[0])
        self.mae = float(losses[1])

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def init_memory(self):
        print("Starting replay memory initialization")
        for i in range(self.mem_init_size):
            self.memory.append(self.data[self.cur_idx])
            self.cur_idx += 1

    def print_stats(self, ep, ep_tot, time, steps):
        if steps == 0:
            steps += 1

        print("\n{}/{}".format(ep, ep_tot),
              "Execution time: {} steps/s".format(round(steps/time, 2)),
              "Mae:", self.mae,
              "Loss:", self.loss
             ) #trail_len

def main(args):
    configs = [
               #Test impact update_freq
               (0.0001, 32, 16, 32, 5000),
               (0.0001, 32, 16, 32, 10000),
               (0.0001, 32, 16, 32, 20000),
               (0.0001, 32, 16, 32, 40000),

               (0.0001, 32, 16, 64, 5000),
               (0.0001, 32, 16, 64, 10000),
               (0.0001, 32, 16, 64, 20000),
               (0.0001, 32, 16, 64, 40000),

               (0.0001, 32, 16, 128, 5000),
               (0.0001, 32, 16, 128, 10000),
               (0.0001, 32, 16, 128, 20000),
               (0.0001, 32, 16, 128, 40000),

                # Test impact LR
               (0.0001, 32, 16, 64, 10000),
               (0.0001, 32, 16, 64, 20000),
               (0.0001, 32, 16, 128, 10000),
               (0.0001, 32, 16, 128, 20000),

               (0.001, 32, 16, 64, 10000),
               (0.001, 32, 16, 64, 20000),
               (0.001, 32, 16, 128, 10000),
               (0.001, 32, 16, 128, 20000),

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
        agent = ModelAgent(data, *exp)
        try:
            agent.init_memory()
        except KeyboardInterrupt:
            print("Interrupted initialization, continuing with training...")

        tot_step = 0

        for ep in tqdm(range(EPISODES)):
            start_time = time.time()
            for step in range(trial_len):
                done = agent.add_sample()
                agent.train_batch() #Internally iterates default (prediction) model
                tot_step += 1  # Used to determine replay memory update

                if tot_step % agent.target_net_update_freq == 0:
                    #print("Updating target network...")
                    agent.target_train()  # iterates target model

                if done:
                    break

            if ep % 2000 == 0:
                agent.print_stats(ep, EPISODES, time.time() - start_time, step)
            metrics['loss'].append(agent.loss)
            metrics['mae'].append(agent.mae)

            if agent.cur_idx >= len(agent.data):
                break

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
    parser.add_argument('-f', type=str, default='train')
    parser.add_argument('-out', type=str, default='train_output')
    args = parser.parse_args()
    main(args)
