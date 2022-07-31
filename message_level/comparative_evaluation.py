import os

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

from disentanglement.direct_timestep_model import train_direct as train_direct_timestep
from disentanglement.message_level import train as message_train
from disentanglement.direct_model import train_direct
from disentanglement.message_level.evaluate import get_reconstruction_error as message_get_reconstruction, f1_loss
from disentanglement.direct_model.direct_eval import get_reconstruction_error as direct_get_reconstruction
from disentanglement.message_level.model_parameters import *
from disentanglement.message_level import load_data

from disentanglement.timestep_level.timestep_evaluate import get_reconstruction_error as timestep_get_reconstruction
from disentanglement.direct_timestep_model.direct_eval import get_reconstruction_error as timestep_direct_get_reconstruction


def getdata_direct(enc,dec,model,data,skip_obs,istimestep):
    if not istimestep:
        r1, labelcat1, predcat1 = message_get_reconstruction(enc, dec, data, skip_obs_agents=skip_obs, shuffleu=False,
                                                             computecat=True)
        r2, labelcat2, predcat2 = direct_get_reconstruction(model, data, computecat=True)
        return r1,r2,labelcat1,labelcat2,predcat1,predcat2
    else:
        r1, labelcat1, predcat1 = timestep_get_reconstruction(enc, dec, data, computecat=True)
        r2, labelcat2, predcat2 = timestep_direct_get_reconstruction(model, data, computecat=True)
        return r1,r2,labelcat1,labelcat2,predcat1,predcat2


def getdata_pair(enc1,dec1,enc2,dec2,data,skip_obs):
    r1, labelcat1, predcat1 = message_get_reconstruction(enc1, dec1, data, skip_obs_agents=skip_obs, shuffleu=False,
                                                         computecat=True)
    r2, labelcat2, predcat2 = message_get_reconstruction(enc2, dec2, data, skip_obs_agents=skip_obs, shuffleu=False,
                                                         computecat=True)
    return r1,r2,labelcat1,labelcat2,predcat1,predcat2

def plot_compare_recontruction_error(alldata,names,filename,skip_obs,timestep=False):
    r1, r2, labelcat1, labelcat2, predcat1, predcat2 = alldata
    for k in r1.keys():
        print("R1 ",k,np.mean(r1[k]))
    for k in r2.keys():
        print("R2 ",k,np.mean(r2[k]))

    rs = [{},{}]

    for k in r1.keys():
        if k in r2.keys():
            rs[0][k] = r1[k]
            rs[1][k] = r2[k]
            a = r1[k]
            b = r2[k]
            print("mean")
            print(k,np.mean(a),np.mean(b))
            print("median")
            print(k,np.median(a),np.median(b))

            print(ttest_ind(a, b, nan_policy='omit'))

    pr = {}
    for i, r in enumerate(rs):
        a = names[i]
        if "Model" not in pr:
            pr["Model"] = np.asarray(len(list(r.values())[0]) * [a])
        else:
            pr["Model"] = np.append(pr["Model"], np.asarray(len(list(r.values())[0]) * [a]))
        for k2 in r.keys():
            r[k2] = np.asarray(r[k2])
            if k2 not in pr:
                pr[k2] = np.asarray(r[k2])
            else:
                pr[k2] = np.append(pr[k2], np.asarray(r[k2]))
    pr2 = {}

    if not timestep:
        namedict = {
            'pos': 'Position',
            'obs_targets': 'Target Obs.',
            'message': 'Message Reconstruction',
            'obs_agents': 'Agent Obs.',
            'Model':'Model'}
        for k in pr.keys():
            if k != "message":
                pr2[namedict[k]] = pr[k]
                print(k, pr[k].shape)
    else:
        namedict = {
            'positions': 'Agent Positions',
            'targets': 'Target Positions',
            'obs_targets': 'Target Obs.',
            'messages': 'Messages',
            'obs_agents': 'Agent Obs.',
            'agent_behavior':"Knowledge Reconstruction",
            'Model': 'Model'}
        order = np.random.permutation(len(pr['targets']))
        for k in pr.keys():
            if k != "message":
                pr2[namedict[k]] = pr[k][order]

    df = pd.DataFrame(pr2)
    df = pd.melt(df, id_vars="Model", value_name="Error")
    plt.rcParams['text.usetex'] = True
    ax = seaborn.boxplot(data=df, x="variable", y="Error", hue="Model", showfliers=False,
                         showmeans=True,
                         hue_order=["Word Knowledge",r"\texttt{MLP-world}"],
                         meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"10"})
    plt.xticks(rotation=30)
    ax.set(title=r'Prediction Error (Euclidean Distance)')
    ax.set(xlabel=None)
    plt.savefig("comparison_errors"+str(filename)+".png",bbox_inches='tight')
    plt.clf()

    if not skip_obs:
        print("Num > 0 agents",torch.sum(labelcat1['obs_agents'].to(torch.float32)))
    print("Num > 0 targets",torch.sum(labelcat1['obs_targets'].to(torch.float32)))

    print("Num total targets",len(labelcat1['obs_targets']))

    vals = ["obs_targets"]
    if not skip_obs:
        vals.append("obs_agents")
    for val in vals:
        f1, precision, recall, accuracy = f1_loss(labelcat1[val], predcat1[val])
        print("Encoder decoder structure:",val)
        print("F1: ",f1)
        print("Precision",precision)
        print("recall",recall)
        print("accuracy",accuracy)

        f1, precision, recall, accuracy = f1_loss(labelcat2[val], predcat2[val])
        print("Direct structure:",val)
        print("F1: ", f1)
        print("Precision", precision)
        print("recall", recall)
        print("accuracy", accuracy)

def plot_comparative_with_direct(type1,num,skip_obs=False):
    modelname1 = "checkpoint_" + str(num - 1) + "_model.pkl"
    logsname1 = "checkpoint_" + str(num - 1) + "_logs.pkl"

    if type1 == "":
        checkpointdir1 = "saved_models/checkpoints/"
    elif type1 == "timestep":
        checkpointdir1 = "../timestep_level/saved_models/checkpoints/"
        logsname1 = "checkpoint" + str(num - 1) + "_logs.pkl"
    else:
        checkpointdir1 = "saved_models/" + type1 + "_checkpoints/"

    _, test_data, _ = load_data.get_data(batch_size=499,agent=(type1 != "timestep"))

    enc, dec, _, _, _, _ = message_train.get_models(checkpointdir1, modelname1, logsname1,load=True, epochs=None, checkpoint_path=checkpointdir1,skip_obs_agents=skip_obs)

    if type1 == "timestep":
        model,_ = train_direct_timestep.get_models("../direct_timestep_model/saved_models", "pickled_model_large_direct.pkl","training_data_large_direct.pkl",None,
                                           load=True)

    else:
        model,_ = train_direct.get_models("../direct_model/saved_models", "pickled_model_large_direct.pkl","training_data_large_direct.pkl",None,
                                           load=True)

    alldata = getdata_direct(enc,dec,model,test_data,skip_obs,istimestep=(type1 == "timestep"))
    plot_compare_recontruction_error(alldata,["Word Knowledge",r"\texttt{MLP-world}"],type1,skip_obs,timestep=True)


def plot_comparative_with_other(type1,num1,type2,num2,names,skip_obs=False):
    if type1 == "":
        checkpointdir1 = "saved_models/checkpoints/"
    else:
        checkpointdir1 = "saved_models/" + type1 + "_checkpoints/"
    modelname1 = "checkpoint_" + str(num1 - 1) + "_model.pkl"
    logsname1 = "checkpoint_" + str(num1 - 1) + "_logs.pkl"

    if type2 == "":
        checkpointdir2 = "saved_models/checkpoints/"
    else:
        checkpointdir2 = "saved_models/" + type2 + "_checkpoints/"
    modelname2 = "checkpoint_" + str(num2 - 1) + "_model.pkl"
    logsname2 = "checkpoint_" + str(num2 - 1) + "_logs.pkl"


    _, test_data, _ = load_data.get_data(batch_size=499)

    enc1, dec1, _, _, _, _ = message_train.get_models(checkpointdir1, modelname1, logsname1,load=True, epochs=None, checkpoint_path=checkpointdir1,skip_obs_agents=skip_obs)
    enc2, dec2, _, _, _, _ = message_train.get_models(checkpointdir2, modelname2, logsname2,load=True, epochs=None, checkpoint_path=checkpointdir2,skip_obs_agents=skip_obs)

    alldata = getdata_pair(enc1,dec1,enc2,dec2,test_data,skip_obs)

    plot_compare_recontruction_error(alldata,names,type1+"_"+type2,skip_obs)


if __name__ == '__main__':
    #seaborn.set(font_scale=1.5)
    plot_comparative_with_direct("timestep",8,skip_obs=False)
    # plot_comparative_with_other("",8,"lbha",9,[r"\texttt{SSDGM}",r"$\beta=0.001,\alpha=100$"],skip_obs=False)

