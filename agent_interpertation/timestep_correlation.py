import os
import pickle
import random

import numpy as np
import pandas as pd
import seaborn
import torch
from matplotlib import pyplot as plt

from disentanglement.direct_timestep_model.direct_model_parameters import *
from disentanglement.direct_timestep_model.train_direct import get_models as get_predictor
from disentanglement.message_level.load_data import get_data
from disentanglement.message_level.utils import alltocuda, ifcuda, alltocpu


def get_distance(a, positions):
    return torch.linalg.norm(positions - a, dim=-1)


def possible_knowledge_indexes(positions, targets):
    output = []
    out_targ = []
    for i, x in enumerate(positions):
        t = targets[i]
        distances = torch.squeeze((get_distance(x[0], x) < 1.3).nonzero(), dim=-1)
        a = []
        b = []
        for d in distances:
            a = np.union1d(a, torch.squeeze((get_distance(x[d], x) < 1.3).nonzero().detach().cpu(), dim=-1)).astype(int)
            b = np.union1d(b, torch.squeeze((get_distance(x[d], t) < 1.3).nonzero().detach().cpu(), dim=-1)).astype(int)
        output.append(a)
        out_targ.append(b)
    return output, out_targ


def correlate_obs_distance(adversary=False):
    predictor, _ = get_predictor("../direct_timestep_model/saved_models", "pickled_model_large_direct.pkl",
                                 "training_data_large_direct.pkl", "large", load=True)
    _, data, _ = get_data(499, save_name='val' if not adversary else 'adversarial', agent=False)
    predictor.eval()
    if CUDA:
        predictor.cuda()

    errors = {
        'obs_agents': [],
        'obs_targets': [],
        'messages': [],
        'positions': [],
        'targets': []
    }
    reals = {
        'obs_agents': [],
        'obs_targets': [],
        'messages': [],
        'positions': [],
        'targets': []
    }
    preds = {
        'obs_agents': [],
        'obs_targets': [],
        'messages': [],
        'positions': [],
        'targets': []
    }

    targets_direct = []
    positions_direct = []

    N = 0

    l = len(data)

    for b, (unprocessed_message, unprocessed) in enumerate(data):
        if b % 100 == 0:
            print(b,"of",l)
        if random.random() < 0.01:
            # Don't look at adversary's worldview
            if adversary:
                adversarymask = torch.where(unprocessed['agent'] != 0, 1, 0)
                indicies = torch.nonzero(adversarymask)
                labels = {}
                message = torch.squeeze(unprocessed_message[indicies],dim=1)
                for k in unprocessed.keys():
                    labels[k] = torch.squeeze(unprocessed[k][indicies],dim=1)
            else:
                message = unprocessed_message
                labels = unprocessed
            alltocuda(labels)

            labels["obs_agents"][labels["obs_agents"] == 0.0] = -1
            labels["obs_targets"][labels["obs_targets"] == 0.0] = -1

            message = ifcuda(message)
            batch_size = message.size()[0]
            N += batch_size
            q = predictor(message)
            possiblepoints, possibletargets = possible_knowledge_indexes(labels['positions'], labels['targets'])

            for i, s in enumerate(possiblepoints):
                t = possibletargets[i]
                poss = np.setdiff1d(s,[0])
                targets_direct.append(
                    torch.flatten(
                        torch.linalg.norm(labels['targets'][i][t] -
                                                          labels['positions'][i][0], dim=-1) <= 1.3
                    ).cpu().detach().numpy()
                )
                reals['targets'].append(torch.linalg.norm(labels['targets'][i][t]-labels['positions'][i][0],dim=-1).cpu().detach().numpy())
                preds['targets'].append(torch.linalg.norm(q['targets'][i][t]-labels['positions'][i][0],dim=-1).cpu().detach().numpy())

                errors['targets'].append(
                    torch.abs(torch.linalg.norm(q['targets'][i][t] - labels['targets'][i][t],dim=-1)).cpu().detach().numpy())

                reals['positions'].append(torch.linalg.norm(labels['positions'][i][poss] - labels['positions'][i][0],
                                                          dim=-1).cpu().detach().numpy()) # Position reals = distance from agent
                preds['positions'].append(
                    torch.linalg.norm(q['positions'][i][poss] - labels['positions'][i][0], dim=-1).cpu().detach().numpy())

                errors['positions'].append(
                    torch.abs(
                        torch.linalg.norm(q['positions'][i][poss] - labels['positions'][i][poss], dim=-1)).cpu().detach().numpy())

                positions_direct.append(
                    torch.flatten(
                       torch.linalg.norm(labels['positions'][i][poss] -
                                                          labels['positions'][i][0], dim=-1) <= 1.3
                    ).cpu().detach().numpy()
                )

                for val in ['obs_agents', 'obs_targets', 'messages',]:
                    errors[val].append(
                        torch.abs(torch.flatten(q[val][i][s] - labels[val][i][s])).cpu().detach().numpy())
                    reals[val].append(torch.flatten(labels[val][i][s]).cpu().detach().numpy())
                    preds[val].append(torch.flatten(q[val][i][s]).cpu().detach().numpy())

            alltocpu(labels)

    for k in errors.keys():
        errors[k] = np.concatenate(errors[k], axis=0)
        reals[k] = np.concatenate(reals[k], axis=0)
        preds[k] = np.concatenate(preds[k], axis=0)

    targets_direct = np.concatenate(targets_direct, axis=0)
    positions_direct = np.concatenate(positions_direct, axis=0)

    return errors, reals, targets_direct, positions_direct, preds


def plot_correlation(adversary=False,load=True):
    savedir = "timestep_results" if not adversary else "adversary_timestep_results"

    if load:
        with open(os.path.join(savedir,"saved_message.pkl"),"rb") as f:
            errors, reals, targets_direct, positions_direct, preds = pickle.load(f)
    else:
        errors, reals, targets_direct, positions_direct, preds = correlate_obs_distance(adversary=adversary)
        with open(os.path.join(savedir,"saved_message.pkl"),'wb') as f:
            pickle.dump((errors, reals, targets_direct, positions_direct, preds),f)

    namedict = {
        'positions': 'Agent Positions',
        'targets': 'Target Positions',
        'obs_targets': 'Target Obs.',
        'messages': 'Messages',
        'obs_agents': 'Agent Obs.',
        'agent_behavior': "Knowledge Reconstruction",
        'Model': 'Model'}

    print(len(errors['positions']))
    for k in errors.keys():
        randorder = np.random.permutation(len(errors[k]))[:100000]
        if k == 'obs_targets' or k == 'obs_agents':
            h = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
            x = {"Real Value": reals[k][randorder],
                 "Error": errors[k][randorder],

                 "Category":
                     np.where(
                         np.logical_and(reals[k][randorder] > 0, preds[k][randorder] > 0),  # if
                         "True Positive",
                         np.where(  # elif
                             np.logical_and(reals[k][randorder] > 0, preds[k][randorder] < 0),
                             "False Negative",
                             np.where(  # elif
                                 np.logical_and(reals[k][randorder] < 0, preds[k][randorder] < 0),
                                 "True Negative",
                                 "False Positive"  # else
                             )
                         ))}
            df = pd.DataFrame(x)
            ax = seaborn.scatterplot(data=df, x="Real Value", y="Error", hue="Category", hue_order=h,alpha=0.3)
            ax.set(title='Real Value vs Error for ' + namedict[k])


        elif k == 'targets':
            h = [True, False]
            x = {"Distance from Agent": reals[k][randorder],
                 "Error": errors[k][randorder],
                 "Direct Observation": targets_direct[randorder]}
            df = pd.DataFrame(x)
            ax = seaborn.jointplot(data=df, x="Distance from Agent", y="Error", hue="Direct Observation",kind="scatter",
                                hue_order=h,alpha=0.3)

        elif k == 'positions':
            h = [True, False]
            x = {"Distance from Agent": reals[k][randorder],
                 "Error": errors[k][randorder],
                 "Direct Observation": positions_direct[randorder]}
            df = pd.DataFrame(x)
            ax = seaborn.jointplot(data=df, x="Distance from Agent", y="Error", hue="Direct Observation", kind="scatter",
                                   hue_order=h,alpha=0.3)

        else:
            x = {"Real Value": reals[k][randorder],
                 "Error": errors[k][randorder]}
            df = pd.DataFrame(x)
            ax = seaborn.scatterplot(data=df, x="Real Value", y="Error")
            ax.set(title='Real Value vs Error for ' + namedict[k])
        plt.savefig(os.path.join(savedir,"error_correlation_" + k + ".png"), bbox_inches='tight')
        plt.clf()


def plot_errors(adversary=False):
    savedir = "timestep_results" if not adversary else "adversary_timestep_results"

    with open(os.path.join(savedir, "saved_message.pkl"), "rb") as f:
        r, reals, targets_direct, positions_direct, preds = pickle.load(f)
    # errors, reals, targets_direct, positions_direct, preds = correlate_obs_distance(adversary=adversary)
    # with open(os.path.join(savedir, "saved_message.pkl"), 'wb') as f:
    #     pickle.dump((errors, reals, targets_direct, positions_direct, preds), f)
    namedict = {
        'positions': 'Agent Positions',
        'targets': 'Target Positions',
        'obs_targets': 'Target Obs.',
        'messages': 'Messages',
        'obs_agents': 'Agent Obs.',
        'agent_behavior': "Knowledge Reconstruction",
        'Model': 'Model'}
    pr = {}
    for k2 in r.keys():
        r[k2] = np.asarray(r[k2])
        if k2 not in pr:
            pr[k2] = np.asarray(r[k2])
        else:
            pr[k2] = np.append(pr[k2], np.asarray(r[k2]))
    pr2 = {}
    order = np.random.permutation(len(pr['targets']))
    for k in pr.keys():
        if k != "message":
            pr2[namedict[k]] = pr[k][order]

    df = pd.DataFrame(pr2)
    df = pd.melt(df, value_name="Error")
    plt.rcParams['text.usetex'] = True
    ax = seaborn.boxplot(data=df, x="variable", y="Error", showfliers=False,
                         showmeans=True,
                         meanprops={"marker": "o",
                                    "markerfacecolor": "white",
                                    "markeredgecolor": "black",
                                    "markersize": "10"})
    plt.xticks(rotation=30)
    ax.set(title=r'Prediction Error (Euclidean Distance)')
    ax.set(xlabel=None)
    plt.savefig("comparison_errors" + str("aaaaa") + ".png", bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    plot_errors(adversary=False)
    # plot_correlation(adversary=False,load=False)
