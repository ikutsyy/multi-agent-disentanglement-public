import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import pandas
import seaborn
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

from disentanglement.libs.probtorch import probtorch
from disentanglement.message_level import load_data, model_parameters
from disentanglement.message_level.evaluate import generate_data, plot_reconstruction_error, get_reconstruction_error
from disentanglement.message_level.train import get_models
from disentanglement.message_level.utils import permute_dims
from load_data import TimestepDataset
from model_parameters import BATCH_SIZE


def plotpositions():

    with open("../../data/params.json", "r") as p:


        params = json.load(p)
        num_agents = params['env_config']['n_agents']
        num_targets = params['env_config']['n_targets']
        obs_size = params['env_config']['obs_vector_size']
        batch_size = BATCH_SIZE
        with open(os.path.join("../../data/medium/info.pkl"), "rb") as f:
            episode_lengths, message_size = pickle.load(f)
            num_agent_elements = sum(episode_lengths)

        eval_episodes = len(episode_lengths)
        our_params = dict(zip(["num_agents", "num_targets", "obs_size", "eval_episodes", "message_size", "batch_size",
                               "num_agent_elements", "episode_lengths"],
                              [num_agents, num_targets, obs_size, eval_episodes, message_size,
                               batch_size, num_agent_elements, episode_lengths]))

    with open(os.path.join("../../data/medium/timestep.pkl"), "rb") as f:
        df = pickle.load(f)
        dataset = TimestepDataset(data=df, params=our_params)
        txs = np.ones(len(dataset)*3)
        tys = np.ones(len(dataset)*3)
        axs = np.ones(len(dataset) * num_agents)
        ays = np.ones(len(dataset) * num_agents)
        agentlist = np.ones(len(dataset) * num_agents)

        for b, (x,y) in enumerate(dataset):
            # print("single-agent world knowledge")
            # print(x.shape)
            # for k in y.keys():
            #     print(k,y[k].shape)
            # break

            bt  = y["targets"]
            for i,target in enumerate(bt):
                txs[3*b+i] = target[0].item()
                tys[3*b+i] = target[1].item()

            bp = y["positions"]
            for j, agent in enumerate(bp):
                axs[num_agents * b + j] = agent[0].item()
                ays[num_agents * b + j] = agent[1].item()
                agentlist[num_agents * b + j] = int(j)

        df2 = pandas.DataFrame({"x":txs,"y":tys})
        print("Made dataframe")
        seaborn.jointplot(data=df2,x="x",y="y")
        plt.figure()
        df3 = pandas.DataFrame({"ax":axs,"ay":ays,"agent":agentlist})
        seaborn.jointplot(data=df3, x="ax", y="ay",hue="agent",palette="bright")
        plt.show()


def plot_recon_diff(name,num):
    modelname = "checkpoint_" + str(num - 1) + "_model.pkl"
    logsname = "checkpoint_" + str(num - 1) + "_logs.pkl"
    # savedir="saved_models/checkpoints"
    savedir = "saved_models/" + name + "checkpoints/"
    skip = "noobs" in savedir or "nothing" in savedir

    enc, dec, discrim, elbo_log, reconstruction_log, discrim_log = get_models(savedir, modelname, logsname,
                                                                              load=True, epochs=0, checkpoint_path=None,
                                                                              skip_obs_agents=False)

    _, test_data, _ = load_data.get_data(batch_size=499)
    rshuff = get_reconstruction_error(enc, dec, test_data, skip_obs_agents=skip, shuffleu=True)
    runshuff, r, d = get_reconstruction_error(enc, dec, test_data, skip_obs_agents=skip, shuffleu=False, ratedist=True)

    print("Rate:",r)
    print("Distortion:",d)

    mshuff = rshuff["message"]
    munshuff = runshuff["message"]

    print("T test:", ttest_ind(mshuff, munshuff, nan_policy='omit', alternative='greater'))

    print("means:")
    print(np.mean(mshuff))
    print(np.mean(munshuff))

    print("medians:")
    print(np.median(mshuff))
    print(np.median(munshuff))

    seaborn.set(font_scale=1.3)

    df = pd.DataFrame(
        {r'$\mathbf{u} \sim q_\phi(\mathbf{u}\mid\mathbf{x}^n)$': munshuff, "$\mathbf{u} \sim p(\mathbf{u})$": mshuff})

    "a" "$mathbf{u} \sim q_\phi(\mathbf{u}\mid\mathbf{x}^n)"
    "b" "$\mathbf{u} \sim p(\mathbf{u})$"

    ax = seaborn.histplot(data=df)
    ax.set(title=r"Reconstruction Error with Conditional and Marginal $\mathbf{u}$")
    ax.set(ylabel="Error")
    plt.savefig(os.path.join("results/", "marginalu_hist_"+name+".png"),
                bbox_inches='tight')
    plt.clf()

    ax = seaborn.boxplot(data=df)
    ax.set(title=r"Reconstruction Error with Conditional and Marginal $\mathbf{u}$")
    ax.set(ylabel="Error")
    plt.savefig(os.path.join("results/", "marginalu_" + name + ".png"),
                bbox_inches='tight')

if __name__ == '__main__':
    plot_recon_diff("",5)
#    plot_recon_diff("noobs_",6)




