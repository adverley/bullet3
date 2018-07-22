import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np

ENV_NAME = "baxterGymEnv"
metrics = ['episode_reward', 'nb_steps', 'mean_absolute_error',
           'loss', 'nb_episode_steps', 'duration', 'episode', 'mean_q']

#New metrics
# metrics = ['episode', 'episode_reward', 'min_reward', 'max_reward', 'mean_q',
#           'max_q', 'min_q', 'mean_action', 'max_action', 'min_action', 'epsilon', 'mae', 'loss']

filepath_experiment = "../../baselines/experiments/"
filepath_plots = "plots/"


def plot(x, y, title, xname, yname, filepath):
    if x is not None:
        plt.plot(x, y)
    else:
        plt.plot(y)

    plt.xlabel(xname)  # 'time (s)'
    plt.ylabel(yname)  # 'voltage (mV)''
    plt.title(title)
    plt.grid(True)
    plt.savefig(filepath)
    plt.show()

def plots(x, *y, title, xname, yname, filepath):

    for i in y:
        plt.plot(x, i[0], label=i[1])

    plt.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)
    plt.ylim(0, 5.0)
    plt.grid(True)
    plt.savefig(filepath)
    plt.show()

def plot_trend_line(x, y, lb, ub, xlabel, ylabel, title, filepath, degree=5):
    plt.plot(x, y, 'o', markersize=.5)
    z = np.polyfit(x, y, degree)
    p = np.poly1d(z)

    if lb is not None and ub is not None:
        zl = np.polyfit(x, lb, degree)
        pl = np.poly1d(zl)
        zu = np.polyfit(x, ub, degree)
        pu = np.poly1d(zu)

    plt.plot(x, p(x), "b")
    plt.fill_between(x, pl(x), pu(x), color='b', alpha=.2)

    plt.xlabel(xlabel)  # 'time (s)'
    plt.ylabel(ylabel)  # 'voltage (mV)''
    plt.title(title)
    plt.savefig(filepath)
    plt.show()

def plot_hist(ub, y, title, xname, yname, filepath):
    n, bins, patches = plt.hist(y, bins=[x+0.5 for x in range(-1,ub)], facecolor='green')
    ax = plt.gca()
    ax.set_xticks([x for x in range(ub)])
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)
    #plt.grid(True)
    #plt.axis([0, 20.5, 0, 2500])
    plt.xlim(0, ub)
    plt.savefig(filepath)
    plt.show()

def loadData(filepath):
    with open(filepath) as json_data:
        data = json.load(json_data)
    return data


def main(args):
    fn = 'baxter_dqn_{}_data.json'.format(args.f)
    #fn = 'baxter_dqn_exp0_config_data.json'
    # Load the data from the json file
    json_filename = os.path.join(
        #filepath_experiment, "ddpg_{}_nocam_log1.json".format(ENV_NAME))
        filepath_experiment, fn)
    data = loadData(json_filename)

    #print(len([400*x - y for x in data['min_reward'] for y in data['episode_reward']]))
    print(len(data['episode_reward']))

    # Create episode reward graph
    title = "Episode reward for DQN"
    xname = "Episode"
    yname = "Reward per Episode"
    plot(data['episode'], data['episode_reward'], title, xname, yname,
         os.path.join(filepath_plots, "dqn_episode_reward.png"))

    try:
        data['completed_step'] = [x if ((x < z) and not y)  else -x for (x,y,z) in data['completion_step']]
        title = "Completion steps for DQN"
        xname = "Episode"
        yname = "Number of steps"
        plot(data['episode'], data['completed_step'], title, xname, yname,
             os.path.join(filepath_plots, "dqn_episode_completion.png"))
    except KeyError:
        print("Can't generated step completion graph")

    # Create episode reward trend graph
    title = "Episode reward trend for DQN"
    xname = "Episode"
    yname = "Reward per Episode"
    plot_trend_line(data['episode'], data['episode_reward'], [x*400 for x in data['min_reward']], [x*400 if x < 1000 else x for x in data['max_reward']],
                    title=title, xlabel=xname, ylabel=yname, filepath=os.path.join(filepath_plots, "dqn_episode_reward_tend.png"))
    # Maybe remove upper and lower bound as they are quite erratic, they only show whether task was completed but
    # this is now shown in the graph above of the completion steps. Set ub and lb to none to remove them

    #Create graph comparing epsilon policy and loss on q-function (for exploration)
    title = 'Exploration policy policy'
    xname = "Episode"
    yname = r'$\epsilon$'

    plots(data['episode'], (data['epsilon'], r'$\epsilon$'), (data['loss'], r'$\mathcal{L}(Q)$'), title=title, xname=xname, yname=yname,
         filepath=os.path.join(filepath_plots, "dqn_episode_epsilon.png"))

    # Create average q_value graph
    title = "Mean Q-value for DQN"
    xname = "Episode"
    yname = "Mean Q-value"
    plot(data['episode'], data['mean_q'], title, xname, yname,
         os.path.join(filepath_plots, "dqn_episode_meanq.png"))

    # Create mae graph
    title = "Mean Absolute Error for DQN"
    xname = "Episode"
    yname = "MAE"
    plot(data['episode'], data['mae'], title, xname, yname,
         os.path.join(filepath_plots, "dqn_episode_mae1.png"))

    # Create loss graph
    title = "Loss for DQN"
    xname = "Episode"
    yname = "Loss"
    plot(data['episode'], data['loss'], title, xname, yname,
         os.path.join(filepath_plots, "dqn_episode_loss1.png"))

    #Create graph with distribution of actions for each episode!!!
    title = "Action distribution for DQN"
    xname = "Episode"
    yname = "Action"
    plot(None, data['mean_action'], title, xname, yname,
         os.path.join(filepath_plots, "dqn_episode_action.png"))

    title = "Action histogram for DQN"
    xname = "Action"
    yname = "Frequency"
    # First find the upper bound of the possible actions
    ub = max(data['max_action'])
    data['action_hist'] = [0 for x in range(ub + 1)]
    for x in data['mean_action']:
        data['action_hist'][int(round(x))] += 1
    print(data['action_hist'])
    plot_hist(ub + 1, [int(round(x)) for x in data['mean_action']], title, xname, yname,
              os.path.join(filepath_plots, "dqn_episode_action_hist.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default=None, help="File name")
    args = parser.parse_args()
    main(args)
