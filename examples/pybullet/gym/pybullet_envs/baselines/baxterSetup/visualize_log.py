import matplotlib.pyplot as plt
import json
import os
import numpy as np
import argparse

ENV_NAME = "baxterGymEnv"
metrics = ['episode_reward', 'nb_steps', 'mean_absolute_error',
           'loss', 'nb_episode_steps', 'duration', 'episode', 'mean_q']

metrics_data = ['step', 'nb_steps', 'episode', 'duration', 'episode_steps',
                'sps', 'episode_reward', 'reward_mean', 'reward_min', 'reward_max',
                'action_mean', 'action_min', 'action_max', 'obs_mean', 'obs_min', 'obs_max', 'metrics']
# Metrics consists of more field which are available in metrics (mean_absolute_error, loss, mean_q)

filepath_experiment = "../experiments/"
filepath_plots = "plots/"


def plot(x, y, title, xname, yname, filepath):
    plt.plot(x, y)

    plt.xlabel(xname)  # 'time (s)'
    plt.ylabel(yname)  # 'voltage (mV)''
    plt.title(title)
    plt.grid(True)
    plt.savefig(filepath)
    plt.show()


def plot_mean_and_CI(x, y, lb, ub, title, xlabel, ylabel, filepath=None):
    X = np.array(x)
    Y = np.array(y)
    error = np.random.normal(0.1, 0.02, size=len(y))

    plt.plot(X, Y)

    # plot the shaded range of the confidence intervals
    plt.fill_between(X, X + error, X - error, alpha=.5)

    plt.xlabel(xlabel)  # 'time (s)'
    plt.ylabel(ylabel)  # 'voltage (mV)''
    plt.title(title)
    plt.savefig(filepath)
    plt.show()


def plot_tend_line(x, y, lb, ub, xlabel, ylabel, title, filepath, degree=5):
    plt.plot(x, y, 'o', markersize=.5)
    z = np.polyfit(x, y, degree)
    p = np.poly1d(z)

    zl = np.polyfit(x, lb, degree)
    pl = np.poly1d(zl)
    zu = np.polyfit(x, ub, degree)
    pu = np.poly1d(zu)

    plt.plot(x, p(x), "b")
    #plt.fill_between(x, pl(x), pu(x), color='b', alpha=.2)

    plt.xlabel(xlabel)  # 'time (s)'
    plt.ylabel(ylabel)  # 'voltage (mV)''
    plt.title(title)
    plt.savefig(filepath)
    plt.show()


def loadData(filepath):
    with open(filepath) as json_data:
        data = json.load(json_data)
    return data


window_length = 1


def main(args):
    # Load the data from the json file
    fn = "baxter_dqn_exp0_config_rand"
    #json_filename = os.path.join(
    #    filepath_experiment, "ddpg_{}_sparse_log.json".format(ENV_NAME))
    #data1 = loadData(json_filename)

    json_filename = os.path.join(
        filepath_experiment, "baxter_dqn_exp0_config_rand_data.json")
    data2 = loadData(json_filename)

    print(data2.keys())

    data2['reward_min'] = [x if x > -400 else -4. for x in data2['reward_min']]

    # Create episode reward graph with trendline
    title = "Mean reward per episode using DDPG without camera images"
    xname = "Episode"
    yname = "Mean reward per episode"
    # plot(data1['episode'], data1['episode_reward'], title, xname, yname,
    #     os.path.join(filepath_plots, "ddpg_episode_reward.png"))
    plot_tend_line(x=data2['episode'][::10],
                   y=data2['reward_mean'][::10],
                   lb=data2['reward_min'][:: 10],
                   ub=data2['reward_max'][:: 10],
                   xlabel=xname,
                   ylabel=yname,
                   title=title,
                   filepath=os.path.join(filepath_plots, "ddpg_episode_reward_trend.png"))

    # Reward plot without trendline
    title = "Mean reward per episode using ddpg without camera images"
    xname = "Episode"
    yname = "Mean reward per episode"
    plot_tend_line(x=data2['episode'][0::10],
                   y=data2['reward_mean'][0::10],
                   lb=data2['reward_min'][0::10],
                   ub=data2['reward_max'][0::10],
                   xlabel=xname,
                   ylabel=yname,
                   title=title,
                   filepath=os.path.join(filepath_plots, "ddpg_episode_reward_trend.png"))

    # Create average q_value graph
    title = "Mean Q-value for ddpg (WINDOW_LENGTH = {})".format(window_length)
    xname = "Episode"
    yname = "Mean Q-value"
    plot(data1['episode'], data1['mean_q'], title, xname, yname,
         os.path.join(filepath_plots, "ddpg_episode_meanq.png"))

    # Create mae graph
    title = "Mean Absolute Error for ddpg (WINDOW_LENGTH = {})".format(
        window_length)
    xname = "Episode"
    yname = "MAE"
    plot(data1['episode'], data1['mean_absolute_error'], title, xname, yname,
         os.path.join(filepath_plots, "ddpg_episode_mae.png"))

    # Create loss graph
    title = "Loss for ddpg (WINDOW_LENGTH = {})".format(window_length)
    xname = "Episode"
    yname = "Loss"
    plot(data1['episode'], data1['loss'], title, xname, yname,
         os.path.join(filepath_plots, "ddpg_episode_loss.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default=None, help="File name")
    args = parser.parse_args()
    main(args)
