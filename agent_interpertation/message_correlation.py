import os
import pickle
import random

import numpy as np
import numpy.random
import pandas as pd
import scipy
import seaborn
import torch
from matplotlib import pyplot as plt
from scipy import stats
from torch import logit
from torch.nn.functional import softmax

from disentanglement.direct_model.direct_model_parameters import *
from disentanglement.direct_model.train_direct import get_models as get_predictor
from disentanglement.message_level.load_data import get_data
from disentanglement.message_level.utils import alltocuda, ifcuda, alltocpu


def correlate_obs_distance(adversary=False):
    predictor, _ = get_predictor("../direct_model/saved_models", "pickled_model_large_direct.pkl",
                                 "training_data_large_direct.pkl", "large", load=True)
    _, data, _ = get_data(499, save_name='val' if not adversary else 'adversarial', agent=False)
    predictor.eval()
    if CUDA:
        predictor.cuda()

    errors = {
        'pos': [],
        'obs_targets': [],
        'obs_agents': [],
    }
    reals = {
        'pos': [],
        'obs_targets': [],
        'obs_agents': [],
    }
    preds = {
        'pos': [],
        'obs_targets': [],
        'obs_agents': [],
    }

    messages = []

    nearest_target_num_covering = []

    N = 0
    l = len(data)
    for b, (_, unprocessed) in enumerate(data):


        if b % 100 == 0:
            print(b, "of", l)
        if random.random() < 0.1:
            # Only look at messages from adversary
            if adversary:
                adversarymask = torch.where(unprocessed['agent'] == 0, 1, 0)
                indicies = torch.nonzero(adversarymask)
                message = torch.sigmoid(unprocessed["messages"][:, 0])[indicies]
                labels = {
                    "pos": torch.squeeze(unprocessed["positions"][:, 0][indicies],1),
                    "obs_agents": torch.squeeze(unprocessed["obs_agents"][:, 0][indicies],1),
                    "obs_targets": torch.squeeze(unprocessed["obs_targets"][:, 0][indicies],1),
                }
            else:
                message = torch.sigmoid(unprocessed["messages"][:, 0])

                labels = {
                    "pos": unprocessed["positions"][:, 0],
                    "obs_agents": unprocessed["obs_agents"][:, 0],
                    "obs_targets": unprocessed["obs_targets"][:, 0],
                }
            messages.append(logit(torch.squeeze(message))[:,:-2])

            labels["obs_agents"][labels["obs_agents"] == 0.0] = -1
            labels["obs_targets"][labels["obs_targets"] == 0.0] = -1


            # Tensor kung-fu
            ntnc = torch.where((torch.linalg.norm(
                unprocessed["targets"][:, 0]- unprocessed["positions"][:,0], dim=-1) <=1.3),
                torch.sum(torch.where(
                torch.linalg.norm(torch.unsqueeze(unprocessed["targets"][:, 0], dim=1) - unprocessed["positions"],
                                  dim=-1) <= 1.3, 1, 0), dim=-1) ,0)

            alltocuda(labels)

            message = ifcuda(message)
            batch_size = message.size()[0]
            N += batch_size
            q = predictor(message)

            errors['pos'].append(torch.abs(torch.flatten(q['pos'] - labels['pos'])).cpu().detach().numpy())

            errors['obs_agents'].append(
                torch.abs(torch.flatten(q['obs_agents'] - labels['obs_agents'])).cpu().detach().numpy())
            errors['obs_targets'].append(
                torch.abs(torch.flatten(q['obs_targets'] - labels['obs_targets'])).cpu().detach().numpy())
            nearest_target_num_covering.append(torch.flatten(
                torch.unsqueeze(ntnc, -1).repeat(1, 16)).cpu().detach().numpy()
                                                 )

            for k in reals.keys():
                reals[k].append(torch.flatten(labels[k]).cpu().detach().numpy())

            for k in preds.keys():
                preds[k].append(torch.flatten(q[k]).cpu().detach().numpy())

            alltocpu(labels)
    for k in errors.keys():
        errors[k] = np.concatenate(errors[k], axis=0)
        reals[k] = np.concatenate(reals[k], axis=0)
        preds[k] = np.concatenate(preds[k], axis=0)
    nearest_target_num_covering = np.concatenate(nearest_target_num_covering, axis=0)
    messages = np.concatenate(messages,axis=0)

    return errors, reals, preds, nearest_target_num_covering, messages


def plot_correlation(adversary=False,load=False,doplots=True):
    savedir = "message_results" if not adversary else "adversary_message_results"

    if load:
        with open(os.path.join(savedir,"saved_message.pkl"),"rb") as f:
            errors, reals, preds, ntna,messages = pickle.load(f)
    else:
        errors, reals, preds, ntna,messages = correlate_obs_distance(adversary=adversary)
        with open(os.path.join(savedir,"saved_message.pkl"),'wb') as f:
            pickle.dump((errors, reals, preds, ntna,messages),f)

    plt.hist(preds["obs_agents"])
    plt.show()

    namedict = {
        'pos': 'Position',
        'obs_targets': 'Target Obs.',
        'message': 'Message Reconstruction',
        'obs_agents': 'Agent Obs.',
        'Model': 'Model'}


    print(len(errors['pos']))

    for k in errors.keys():
        if doplots:
            randorder = np.random.permutation(len(reals[k]))[:100000]
        else:
            randorder = np.random.permutation(len(reals[k]))

        if k == 'pos':
            x = {"Real Value": reals[k][randorder],
                 "Error": errors[k][randorder]}
            df = pd.DataFrame(x)

            if doplots:
                ax = seaborn.scatterplot(data=df, x="Real Value", y="Error")
                ax.set(title='Real Value vs Error for ' + namedict[k])

        elif k == 'obs_targets':
            h = ['True Negative', 'False Positive', 'True Positive', 'False Negative',]
            x = {"Real Value": reals[k][randorder],
                 "Error": errors[k][randorder],
                 "Near a Target": ntna[randorder] > 0,
                 "Two Agents Near Closest Target": ntna[randorder]>1,
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

            print(k)
            print("Two agents near:")

            x = df.loc[(df["Category"] == "False Negative")]["Real Value"]
            y = df.loc[(df["Category"] == "False Negative")]["Error"]

            print("targets",stats.linregress(x, y))


            fna = len(df.loc[(df["Category"]=="False Negative") & (df["Two Agents Near Closest Target"] == True)])
            tpa = len(df.loc[(df["Category"]=="True Positive") & (df["Two Agents Near Closest Target"] == True )])

            print(fna,tpa,"Recall when two agents near:",tpa/(fna+tpa))

            fnb = len(df.loc[(df["Category"]=="False Negative") & (df["Two Agents Near Closest Target"] == False)])
            tpb = len(df.loc[(df["Category"]=="True Positive") & (df["Two Agents Near Closest Target"] == False)])
            print(fnb,tpb,"Recallwhen two agents  NOT near:",tpb/(fnb+tpb))


            if doplots:
                ax = seaborn.relplot(data=df, x="Real Value", y="Error", hue="Category",alpha=0.3,
                                     hue_order=h, kind='scatter')
                # plt.suptitle('Real Value vs Error for ' + namedict[k])

        else:
            h = [ 'True Negative', 'False Positive', 'False Negative','True Positive']
            x = {"Real Value": reals[k][randorder],
                 "Error": errors[k][randorder],
                 "Near a Target": ntna[randorder] > 0,
                 "Two Agents Near Closest Target": ntna[randorder] > 1,
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

            x = df.loc[(df["Category"] == "False Negative")]["Real Value"]
            y = df.loc[(df["Category"] == "False Negative")]["Error"]

            print("agents", stats.linregress(x, y))

            print(k)

            fna = len(df.loc[(df["Category"] == "False Negative") & (df["Two Agents Near Closest Target"] == True)])
            tpa = len(df.loc[(df["Category"] == "True Positive") & (df["Two Agents Near Closest Target"] == True)])

            print(fna, tpa, "Recall when two agents near:", tpa / (fna + tpa))

            fnb = len(df.loc[(df["Category"] == "False Negative") & (df["Two Agents Near Closest Target"] == False)])
            tpb = len(df.loc[(df["Category"] == "True Positive") & (df["Two Agents Near Closest Target"] == False)])
            print(fnb, tpb, "Recall when two agents NOT near:", tpb / (fnb + tpb))

            print(k)
            print("We are near:")

            fna = len(df.loc[(df["Category"] == "False Negative") & (df["Near a Target"] == True)])
            tpa = len(df.loc[(df["Category"] == "True Positive") & (df["Near a Target"] == True)])

            print(fna, tpa, "Recall when we are near:", tpa / (fna + tpa))

            fnb = len(df.loc[(df["Category"] == "False Negative") & (df["Near a Target"] == False)])
            tpb = len(df.loc[(df["Category"] == "True Positive") & (df["Near a Target"] == False)])
            print(fnb, tpb, "Recall when we are NOT near:", tpb / (fnb + tpb))

            if doplots:
                ax = seaborn.relplot(data=df, x="Real Value", y="Error", hue="Category",alpha=0.3,
                                     hue_order=h, kind='scatter')
                # plt.suptitle('Real Value vs Error for ' + namedict[k])


        if doplots:
            plt.savefig(os.path.join(savedir,"error_correlation_" + k + ".png"), bbox_inches='tight')
            plt.clf()

def plot_message_histograms():
    with open(os.path.join("message_results", "saved_message.pkl"), "rb") as f:
        _, _, _, _, normal_messages = pickle.load(f)

    with open(os.path.join("adversary_message_results", "saved_message.pkl"), "rb") as f:
        _, _, _, _, adversary_messages = pickle.load(f)

    print(len(normal_messages))
    print(len(adversary_messages))

    l = min(min(len(normal_messages),len(adversary_messages)),1000)
    print("Using",l,"values")
    numpy.random.shuffle(normal_messages) # Only on first dimension
    numpy.random.shuffle(adversary_messages)
    normal_messages = normal_messages[:l]
    adversary_messages = adversary_messages[:l]




    print("Running tests")
    for i in range(64):
        print(stats.kstest(normal_messages[:,i],adversary_messages[:,i]))

    df = pd.DataFrame(np.concatenate([normal_messages],axis=0),columns=range(64))
    df = pd.melt(df)
    df["Source"] = ["Cooperative"]*l*64

    df2 = pd.DataFrame(np.concatenate([adversary_messages],axis=0),columns=range(64))
    df2 = pd.melt(df2)


    df2["Source"] = ["Adversarial"]*l*64

    dff = pd.concat([df,df2], ignore_index=True)

    dff["row"] = dff["variable"]%8
    dff["col"] = dff["variable"]//8

    dff = dff.reset_index(drop=True)
    print(dff)

    seaborn.displot(
        data=dff, x="value",hue="Source", col="col", row="row",kind="kde"
    )

    plt.show()





if __name__ == '__main__':
    #plot_message_histograms()
    plot_correlation(adversary=True,load=False,doplots=True)
