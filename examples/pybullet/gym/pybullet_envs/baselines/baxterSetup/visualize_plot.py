import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np
import math

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

    plt.rcParams.update({'font.size': 12})
    #plt.gcf().subplots_adjust(left=0.17)
    plt.tight_layout(pad=1.8)
    plt.xlabel(xname)  # 'time (s)'
    plt.ylabel(yname)  # 'voltage (mV)''
    plt.xlim(min(min(x), 0), max(max(x), 10000))
    plt.ylim(min(min(y), 0)*1.1, max(y)*1.1)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filepath)
    plt.show()

def plots(x, *y, title, xname, yname, filepath):

    y_min = 0
    y_max = 0
    for i in y:
        plt.plot(x, i[0], label=i[1])
        y_min = min(i[0]) if min(i[0]) < 0 else y_min
        y_max = max(i[0]) if max(i[0]) > 0 else y_max

    # ax2 = ax1.twinX() for second axis
    plt.rcParams.update({'font.size': 12})
    plt.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)
    #plt.ylim(y_min*1.1, y_max*1.1)
    plt.xlim(min(min(x), 0), max(max(x), 10000))
    #plt.ylim(min(min(y), 0), max(y)*1.2)
    plt.ylim(0, 7.0)
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
    if lb is not None and ub is not None:
        plt.fill_between(x, pl(x), pu(x), color='b', alpha=.2)

    plt.rcParams.update({'font.size': 12})
    #plt.gcf().subplots_adjust(left=0.15)
    plt.tight_layout(pad=1.8)
    plt.xlabel(xlabel)  # 'time (s)'
    plt.ylabel(ylabel)  # 'voltage (mV)''
    plt.title(title)
    plt.xlim(min(min(x), 0), max(max(x), 10000))
    plt.grid(True)
    plt.savefig(filepath)
    plt.show()

def plot_hist(ub, y, title, xname, yname, filepath, zoneHist=False, colors=None, labels=None):
    #n, bins, patches = plt.hist(y, bins=[x+1 for x in range(-1,ub)], color='g', label=labels, rwidth=0.5) #facecolor='green',
    if zoneHist:
        if colors is None:
            colors = ['0.55', '#ffff66', '#ff3333', '#00b33c']
        if labels is None:
            labels = ['Outside', 'Cone', 'Torus collision', 'Completion']
        bins = [0, 4]
        plt.hist(y, bins=bins, color=colors, label=labels) #facecolor='green',
        plt.rcParams.update({'font.size': 12})
        ax = plt.gca()
        ax.set_xticklabels([])
        plt.xlim(0, ub)
        plt.legend()
    else:
        bins = list(set(y))
        bins = [x+y for x in bins for y in (-0.5, 0.5)]
        print(bins)
        #bins = [x + 1 for x in range(-1, ub)]
        plt.hist(y, bins=bins, color='g', label=labels, rwidth=0.5) #facecolor='green',
        plt.rcParams.update({'font.size': 12})
        ax = plt.gca()
        plt.xlim(min(y)-1, ub)
        plt.xlabel(xname)

    ax = plt.gca()
    ax.set_xticks([x for x in range(ub)])
    #plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)
    #plt.grid(True)
    #plt.axis([0, 20.5, 0, 2500])
    plt.savefig(filepath)
    plt.show()

def bar_plot(labels, y, title, xname, filepath, colors):
    if colors is None:
        colors = ['0.55', '#ffff66', '#ff3333', '#00b33c']
    fig, ax = plt.subplots()
    plt.bar(range(len(labels)), y, color=colors) # Needs to be changed if more than 4 actions are present
    plt.xticks(range(len(labels)), labels, color='0', style='oblique')
    #plt.yticks(color='0.2', family='cursive', style='italic')
    #plt.xlabel(xname, size='medium', style='oblique')
    plt.title(title, style='oblique')
    plt.savefig(filepath)
    plt.show()

def plot_zoneSuccess(x, y, zones, title, xname, yname, filepath):
    fig, ax = plt.subplots()
    ax.plot(x, y, color='black', linewidth=1)

    colors = ['#FFF2CC', '#F8CECC', '#DAE8FC', '#D5E8D4', '#E1D5E7', '#EEEEEE']

    if len(zones)-1 != len(colors):
       #colors = ['red' if i%2==0 else 'green' for i in range(len(zones)-1)]
       colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231',
                 '#911eb4', '#46f0f0', '#d2f53c', '#008080', '#e6beff', '#fffac8']
       alpha=.2
    else:
        # Put zone labels for exp36
        ax.text(34500, 0.19, "Zone 3", color='#132112')
        ax.text(80500, 0.19, "Zone 5", color='#1a1a1a')
        alpha=1.

    for i in range(len(zones)-1):
        if i != (len(zones) - 2):
            ax.axvspan(zones[i], zones[i+1], alpha=alpha, color=colors[i])
        else:
            ax.axvspan(zones[i], max(x), alpha=alpha, color=colors[i])


    plt.rcParams.update({'font.size': 12})
    #plt.gcf().subplots_adjust(left=0.17)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_xlim(min(min(x), 0), max(max(x), 10000))
    ax.set_ylim(min(min(y), 0)*1.1, max(y)*1.1)
    ax.set_title(title)
    ax.grid()
    plt.savefig(filepath)
    plt.show()

def plot_exploration_polixy(x, y1, y2, title, xname, yname, filepath):

    y_min = 0
    y_max = 0
    for i in (y1, y2):
        y_min = min(i) if min(i) < 0 else y_min
        y_max = max(i) if max(i) > 0 else y_max

    plt.rcParams.update({'font.size': 12})
    plt.legend()
    plt.title(title)
    #plt.grid(True)

    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, '#1F77B4')
    ax1.set_xlabel(xname)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel(yname[0], color='#1F77B4', rotation='horizontal', labelpad=20)
    ax1.tick_params('y', colors='#1F77B4')
    ax1.grid(alpha=.4, color='#1F77B4')
    ax1.set_xlim([0, max(x)])
    #ax1.set_ylim([0, 1.2])

    ax2 = ax1.twinx()
    ax2.plot(x, y2, '#FF7F0E')
    ax2.set_ylabel(yname[1], color='#FF7F0E', rotation='horizontal', labelpad=20)
    ax2.tick_params('y', colors='#FF7F0E')
    ax2.grid(alpha=.4, color='#FF7F0E')
    ax2.set_xlim([0, max(x)])
    ax2.set_ylim([0, 7.0])

    fig.tight_layout()

    #plt.ylim(y_min*1.1, y_max*1.1)
    #plt.xlim(min(min(x), 0), max(max(x), 10000))
    #plt.ylim(min(min(y), 0), max(y)*1.2)
    #plt.ylim(0, 7.0)
    plt.savefig(filepath)
    plt.show()

def loadData(filepath):
    with open(filepath) as json_data:
        data = json.load(json_data)
    return data


def main(args):
    fn = 'baxter_dqn_{}_data.json'.format(args.f)
    test_fn = 'baxter_dqn_{}_testData.json'.format(args.f)
    #fn = 'baxter_dqn_exp0_config_data.json'
    # Load the data from the json file
    json_filename = os.path.join(
        #filepath_experiment, "ddpg_{}_nocam_log1.json".format(ENV_NAME))
        filepath_experiment, fn)
    data = loadData(json_filename)

    test_data = None
    try:
        json_fn = os.path.join(filepath_experiment, test_fn)
        test_data = loadData(json_fn)
        print("Loaded test data!")
    except FileNotFoundError:
        print("Test data not found!")

    #print(len([400*x - y for x in data['min_reward'] for y in data['episode_reward']]))
    print(len(data['episode_reward']))

    # Create episode reward graph
    title = "Episode reward for DQN"
    xname = "Episode"
    yname = "Reward per Episode"
    plot(data['episode'], data['episode_reward'], title, xname, yname,
         os.path.join(filepath_plots, "dqn_episode_reward.png"))

    try:
        data['completed_step'] = [x if ((x < z-1) and not y)  else -x for (x,y,z) in data['completion_step']]
        title = "Completion steps for DQN"
        xname = "Episode"
        yname = "Number of steps"
        #print(len([1 if x['completion_step']]))
        plot(data['episode'], data['completed_step'], title, xname, yname,
             os.path.join(filepath_plots, "dqn_episode_completion.png"))
    except KeyError:
        print("Can't generated step completion graph")

    try:
        # Create graph of success rate
        title = "Success rate"
        xname = "Episode"
        yname = "Success rate"
        succes_data = [[] for x in range(int(math.ceil(len(data['completed_step'])/100)))]

        count = 0
        j = 0
        for i in data['completion_step']:
            if count == 100:
                j += 1
                count = 0
            succes_data[j].append((i[0] < (i[2]-1)) and not i[1])
            count += 1

        #print("Succes data:", succes_data)
        #succes_data = [np.array(x) for x in succes_data]
        #plot(data['episode'][::100], [np.average(x) for x in succes_data], title, xname, yname,
        #     os.path.join(filepath_plots, "dqn_episode_succes_rate.png"))
        plot_zoneSuccess(data['episode'][::100], [np.average(x) for x in succes_data],
                         zones=[data['episode'][i] for i in range(len(data['episode'])) if data['epsilon'][i] - data['epsilon'][i-1] >= 0.8],
                         title=title, xname=xname, yname=yname, filepath= os.path.join(filepath_plots, "dqn_episode_succes_rate.png"))
    except KeyError:
        print("Can't generate success rate graph")

    # Create episode reward trend graph
    title = "Episode reward trend"
    xname = "Episode"
    yname = "Reward per Episode"
    lb = [x*400 for x in data['min_reward']]
    ub = [x*400 if x < 1000 else x for x in data['max_reward']]
    plot_trend_line(data['episode'], data['episode_reward'], None, None,
                    title=title, xlabel=xname, ylabel=yname, filepath=os.path.join(filepath_plots, "dqn_episode_reward_tend.png"))
    # Maybe remove upper and lower bound as they are quite erratic, they only show whether task was completed but
    # this is now shown in the graph above of the completion steps. Set ub and lb to none to remove them

    try:
        #Create graph comparing epsilon policy and loss on q-function (for exploration)
        title = 'Exploration policy'
        xname = "Episode"
        yname = r'$\epsilon$/loss'

        #plots(data['episode'], (data['epsilon'], r'$\epsilon$'), (data['loss'], r'$\mathcal{L}(Q)$'), title=title, xname=xname, yname=yname,
        #     filepath=os.path.join(filepath_plots, "dqn_episode_epsilon.png"))
        plot_exploration_polixy(data['episode'], data['epsilon'], data['loss'], title=title, xname=xname, yname=['$\epsilon$', '$\mathcal{L}(Q)$'],
                                filepath=os.path.join(filepath_plots, "dqn_episode_epsilon.png"))

        # Create average q_value graph
        title = "Mean Q-value for DQN"
        xname = "Episode"
        yname = "Mean Q-value"
        plot(data['episode'], data['mean_q'], title, xname, yname,
             os.path.join(filepath_plots, "dqn_episode_meanq.png"))
    except KeyError:
        print("Can't generate Loss graph")

    try:
        # Create mae graph
        title = "Mean Absolute Error for DQN"
        xname = "Episode"
        yname = "MAE"
        plot(data['episode'], data['mae'], title, xname, yname,
             os.path.join(filepath_plots, "dqn_episode_mae1.png"))
    except KeyError:
        print("Can't generate Loss graph")

    try:
        # Create loss graph
        title = "Network loss"
        xname = "Episode"
        yname = "Loss"
        plot(data['episode'], data['loss'], title, xname, yname,
             os.path.join(filepath_plots, "dqn_episode_loss1.png"))

        print("Over 200:", len([x for x in data['loss'] if x > 200]))
        print("Smaller than 100:", len([x for x in data['loss'] if x < 100]))
        print("Total length:", len(data['loss']))
    except KeyError:
        print("Can't generate Loss graph")

    # Zone steps for completely random
    # TODO: if title contains random plot this graph
    try:
        title = "Episodes in each zone for DQN"
        xname = "Zones"
        yname = ""

        ub = len(data['zone_steps'][0])
        zone_info = {'outside': 0, 'cone': 0, 'torus_col': 0, 'completion': 0}
        hist_data = [[], [], [], []]
        for x in data['zone_steps']:
            if x['outside'] > 0:
                zone_info['outside'] += 1
                hist_data[0].append(0)
            if x['cone'] > 0:
                zone_info['cone'] += 1
                hist_data[1].append(1)
            if x['torus_col'] > 0:
                zone_info['torus_col'] += 1
                hist_data[2].append(2)
            if x['completion'] > 0:
                zone_info['completion'] += 1
                hist_data[3].append(3)

        print("Zone info:", zone_info)

        # TODO: Create graph such that label of zone is shown below histogram
        # plot_hist(ub, hist_data, title, xname, yname,
        #           os.path.join(filepath_plots, "dqn_episode_zones.png"),
        #           zoneHist=True)
        bar_plot(['Outside', 'Cone', 'Torus collision', 'Completion'], [len(x) for x in hist_data],
                 title, xname, os.path.join(filepath_plots, "dqn_episode_zones.png"), None)
    except KeyError:
        print("Can't generate zone graph")

    # Batch steps for completely random
    # TODO: if title contains random plot this graph
    try:
        title = "Zones seen during batch training" #For DQN
        xname = "Zones"
        yname = ""

        ub = len(data['zone_batch'][0])
        zone_info = {'outside': 0, 'cone': 0, 'torus_col': 0, 'completion': 0}
        hist_data = [[], [], [], []]

        avg_outside = np.average([x['outside'] for x in data['zone_batch']])
        avg_cone = np.average([x['cone'] for x in data['zone_batch']])
        avg_torus_col = np.average([x['torus_col'] for x in data['zone_batch']])
        avg_completion = np.average([x['completion'] for x in data['zone_batch']])

        for x in range(int(avg_outside)):
            zone_info['outside'] += 1
            hist_data[0].append(0)

        for x in range(int(avg_cone)):
            zone_info['cone'] += 1
            hist_data[1].append(1)

        for x in range(int(avg_torus_col)):
            zone_info['torus_col'] += 1
            hist_data[2].append(2)

        for x in range(int(avg_completion)):
            zone_info['completion'] += 1
            hist_data[3].append(3)

        # for x in data['zone_batch']:
        #     if x['outside'] > 0:
        #         zone_info['outside'] += 1
        #         hist_data[0].append(0)
        #     if x['cone'] > 0:
        #         zone_info['cone'] += 1
        #         hist_data[1].append(1)
        #     if x['torus_col'] > 0:
        #         zone_info['torus_col'] += 1
        #         hist_data[2].append(2)
        #     if x['completion'] > 0:
        #         zone_info['completion'] += 1
        #         hist_data[3].append(3)

        print("Zone info:", zone_info)

        # TODO: Create graph such that label of zone is shown below histogram
        # plot_hist(ub, hist_data, title, xname, yname,
        #           os.path.join(filepath_plots, "dqn_batch_zones.png"),
        #           zoneHist=True)
        bar_plot(['Outside', 'Cone', 'Torus collision', 'Completion'], [len(x) for x in hist_data],
                 title, xname, os.path.join(filepath_plots, "dqn_episode_batch_zones.png"), None)
    except KeyError:
        print("Can't generate batch zone graph")

    #Create graph with distribution of actions for each episode!!!
    title = "Action distribution"
    xname = "Episode"
    yname = "Action"
    plot(data['episode'], data['mean_action'], title, xname, yname,
         os.path.join(filepath_plots, "dqn_episode_action.png"))

    title = "Action histogram for DQN"
    xname = "Action"
    yname = "Frequency"
    # First find the upper bound of the possible actions
    ub = max(data['max_action'])
    data['action_hist'] = [0 for x in range(ub + 1)]
    for x in data['mean_action']:
        if int(round(x)) > 5:
            idx = data['mean_action'].index(x)
            try:
                #print(x, data['max_action'][idx], data['completion_step'][idx])
                pass
            except KeyError:
                pass
        data['action_hist'][min(int(round(x)), ub)] += 1
    print(data['action_hist'])
    plot_hist(ub + 1, [int(round(x)) for x in data['mean_action']], title, xname, yname,
              os.path.join(filepath_plots, "dqn_episode_action_hist.png"))

    if test_data is not None:
        title = "Action histogram (test data)" #(Test data)
        xname = "Action"
        yname = "Frequency"

        ub = max([max(x) for x in test_data])
        hist_data = [[] for i in range(ub+1)]

        for i in range(ub+1):
            for j in range(len(test_data)):
                hist_data[i].append(test_data[j].count(i))

        labels = ['left', 'right', 'upward', 'downward']
        bar_plot(labels, np.average(hist_data, axis=1), title, xname,
                 filepath=os.path.join(filepath_plots, "dqn_test_action_hist.png"), colors='g')
    else:
        print("No test data for action histogram!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default=None, help="File name")
    args = parser.parse_args()
    main(args)
