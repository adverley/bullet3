from rl.callbacks import Callback

import warnings
import timeit
import json
from tempfile import mkdtemp

import numpy as np

from keras import __version__ as KERAS_VERSION
from keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList
from keras.utils.generic_utils import Progbar


class DataLogger(Callback):
    def __init__(self, filepath):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dictionary that is indexed by the episode to separate episodes
        # from each other.
        self.filepath = filepath

        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0
        self.data = {}

    def on_train_begin(self, logs):
        """ Print training values at beginning of training """
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        #print('Training for {} steps ...'.format(self.params['nb_steps']))

    def on_train_end(self, logs):
        """ Print training time at end of training """
        duration = timeit.default_timer() - self.train_start
        #print('done, took {:.3f} seconds'.format(duration))

        """ Save model at the end of training """
        self.save_data()

    def on_episode_begin(self, episode, logs):
        """ Reset environment variables at beginning of each episode """
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs):
        """ Compute and print training statistics of the episode when done """
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        # Format all metrics.
        metrics = np.array(self.metrics[episode])
        metrics_template = ''
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                if idx > 0:
                    metrics_template += ', '
                try:
                    value = np.nanmean(metrics[:, idx])
                    metrics_template += '{}: {:f}'
                except Warning:
                    value = '--'
                    metrics_template += '{}: {}'
                metrics_variables += [name, value]
        metrics_text = metrics_template.format(*metrics_variables)

        nb_step_digits = str(
            int(np.ceil(np.log10(self.params['nb_steps']))) + 1)
        template = '{step: ' + nb_step_digits + \
            'd}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps}, steps per second: {sps:.0f}, episode reward: {episode_reward:.3f}, mean reward: {reward_mean:.3f} [{reward_min:.3f}, {reward_max:.3f}], mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}], mean observation: {obs_mean:.3f} [{obs_min:.3f}, {obs_max:.3f}], {metrics}'
        variables = {
            'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            'obs_mean': np.mean(self.observations[episode]),
            'obs_min': np.min(self.observations[episode]),
            'obs_max': np.max(self.observations[episode]),
            'metrics': metrics_text,
        }
        # print(template.format(**variables))

        for key, value in variables.iteritems():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1

    def save_data(self):
        """ Save metrics in a json file """
        if len(self.data.keys()) == 0:
            return

        # Sort everything by episode.
        assert 'episode' in self.data
        sorted_indexes = np.argsort(self.data['episode'])
        sorted_data = {}
        for key, values in self.data.items():
            assert len(self.data[key]) == len(sorted_indexes)
            # We convert to np.array() and then to list to convert from np datatypes to native datatypes.
            # This is necessary because json.dump cannot handle np.float32, for example.
            sorted_data[key] = np.array(
                [self.data[key][idx] for idx in sorted_indexes]).tolist()

        # Overwrite already open file. We can simply seek to the beginning since the file will
        # grow strictly monotonously.
        with open(self.filepath, 'w') as f:
            json.dump(sorted_data, f)
