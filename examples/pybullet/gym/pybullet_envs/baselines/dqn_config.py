import json

# Load this from a file instead of putting it here (like config in example)
config_file = 'baxter_dqn_config.json'

class DQNConfig():
    def __init__(self):
        self.experiments = experiments

    def load_config(self):
        with open(config_file, 'r') as fp:
            experiments = json.load(fp)

    def save_config(self):
        with open(config_file, 'w') as fp:
            json.dump(experiments, fp)

    def add_field(field_name):
        for d in experiments:
            d[field_name] = None

    def add_experiment(item):
        learning_rate = [0.05, 0.0025]
        memory_size = [2000]
        gamma = [0.99, 0.85]
        replay_mem_update_freq = [10000, 20000, 50000]
        replay_mem_init_size = [50000]
        loss_function = ['mse', 'huber']
        optimizer = ['adam', 'RMSProb']
        experiments.append(item)

    def print_config(self):
        print(self.experiments)

experiments = [
    {'name': 'exp0',
     'learning_rate' : 0.05,
     'memory_size': 200000,
     'gamma': 0.99,
     'replay_mem_update_freq': 10000,
     'replay_mem_init_size': 50000,
     'loss_function': 'mse',
     'optimizer': 'adam',
     'reward': 'line_distance',
     'randomPos': True,
     'epsilon_start': 500,
     'epsilon_decay': 0.995,
     'epsilon_guided': False
     },
    {'name': 'exp1',
     'learning_rate' : 0.05,
     'memory_size': 200000,
     'gamma': 0.99,
     'replay_mem_update_freq': 10000,
     'replay_mem_init_size': 50000,
     'loss_function': 'mse',
     'optimizer': 'adam',
     'reward': 'torus_distance',
     'randomPos': True,
     'epsilon_start': 500,
     'epsilon_decay': 0.995,
     'epsilon_guided': False
     },
    {'name': 'exp2',
     'learning_rate' : 0.05,
     'memory_size': 200000,
     'gamma': 0.99,
     'replay_mem_update_freq': 10000,
     'replay_mem_init_size': 50000,
     'loss_function': 'huber',
     'optimizer': 'RMSprob',
     'reward': 'line_distance',
     'randomPos': True,
     'epsilon_start': 500,
     'epsilon_decay': 0.995,
     'epsilon_guided': False
     },
    {'name': 'exp3',
     'learning_rate': 0.05,
     'memory_size': 200000,
     'gamma': 0.99,
     'replay_mem_update_freq': 10000,
     'replay_mem_init_size': 50000,
     'loss_function': 'huber',
     'optimizer': 'RMSprob',
     'reward': 'torus_distance',
     'randomPos': True,
     'epsilon_start': 500,
     'epsilon_decay': 0.995,
     'epsilon_guided': False
     },
    {'name': 'exp4',
     'learning_rate' : 0.05,
     'memory_size': 200000,
     'gamma': 0.85,
     'replay_mem_update_freq': 10000,
     'replay_mem_init_size': 50000,
     'loss_function': 'mse',
     'optimizer': 'adam',
     'reward': 'line_distance',
     'randomPos': True,
     'epsilon_start': 500,
     'epsilon_decay': 0.995,
     'epsilon_guided': False
     },
    {'name': 'exp5',
     'learning_rate' : 0.0025,
     'memory_size': 200000,
     'gamma': 0.99,
     'replay_mem_update_freq': 20000,
     'replay_mem_init_size': 50000,
     'loss_function': 'mse',
     'optimizer': 'adam',
     'reward': 'line_distance',
     'randomPos': True,
     'epsilon_start': 500,
     'epsilon_decay': 0.995,
     'epsilon_guided': False
     },
    {'name': 'exp6',
     'learning_rate' : 0.0025,
     'memory_size': 200000,
     'gamma': 0.99,
     'replay_mem_update_freq': 50000,
     'replay_mem_init_size': 50000,
     'loss_function': 'mse',
     'optimizer': 'adam',
     'reward': 'line_distance',
     'randomPos': True,
     'epsilon_start': 500,
     'epsilon_decay': 0.995,
     'epsilon_guided': False
     },
    {'name': 'exp7',
     'learning_rate' : 0.0025,
     'memory_size': 200000,
     'gamma': 0.99,
     'replay_mem_update_freq': 10000,
     'replay_mem_init_size': 50000,
     'loss_function': 'mse',
     'optimizer': 'adam',
     'reward': 'line_distance',
     'randomPos': True,
     'epsilon_start': 500,
     'epsilon_decay': 0.995,
     'epsilon_guided': False
     },
    {'name': 'exp8',
     'learning_rate' : 0.05,
     'memory_size': 200000,
     'gamma': 0.99,
     'replay_mem_update_freq': 10000,
     'replay_mem_init_size': 50000,
     'loss_function': 'mse',
     'optimizer': 'adam',
     'reward': 'torus_distance',
     'randomPos': True,
     'epsilon_start': 1000,
     'epsilon_decay': 0.9977,
     'epsilon_guided': False
     },
    {'name': 'exp9',
     'learning_rate' : 0.05,
     'memory_size': 200000,
     'gamma': 0.99,
     'replay_mem_update_freq': 10000,
     'replay_mem_init_size': 50000,
     'loss_function': 'mse',
     'optimizer': 'adam',
     'reward': 'line_distance',
     'randomPos': True,
     'epsilon_start': 1000,
     'epsilon_decay': 0.9977,
     'epsilon_guided': False
     }
]