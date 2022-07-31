import os
import pickle

import numpy as np
import pandas as pd
import seaborn as seaborn
import sklearn
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import accuracy_score, precision_score, f1_score

from disentanglement.message_level import train, load_data
from disentanglement.message_level.load_data import get_data
from disentanglement.message_level.model_parameters import *
from disentanglement.message_level.train import get_models, test


def generate_data(enc, interested_values, test_data=None, distributions=False, both=False):
    if test_data is None:
        batch_size = BATCH_SIZE
        _, test_data, _ = load_data.get_data(batch_size)

    result = dict(zip(interested_values, [None] * len(interested_values)))
    if both:
        resultmeans = dict(zip(interested_values, [None] * len(interested_values)))
    for b, (message, labels) in enumerate(test_data):
        if CUDA:
            message = message.cuda()
        q, dist = enc(message, num_samples=1, extract_distributions=True)

        for value in interested_values:
            if distributions:
                z = dist[value].cpu().detach().numpy()
            elif both:
                z = q[value].value.squeeze().cpu().detach().numpy()
                z2 = dist[value].cpu().detach().numpy()

            else:
                z = q[value].value.squeeze().cpu().detach().numpy()
            if result[value] is None:
                result[value] = z
                if both:
                    resultmeans[value] = z2
            else:
                result[value] = np.append(result[value], z, axis=0)
                if both:
                    resultmeans[value] = np.append(resultmeans[value], z2, axis=0)
    if both:
        return result, resultmeans
    return result


def plot_z_embeddings(enc, Zs=None, dofig=True, stylename='z', title=""):
    if Zs is None:
        Zs = generate_data(enc, [stylename])[stylename]

    df = pd.DataFrame(Zs, columns=[r'$\mathbf{stylename{%d}}$' % i for i in range(Zs.shape[1])])
    p = seaborn.pairplot(df)
    # figs.suptitle(r'$Z \ Embeddings$', fontsize=30)
    # axes[Z_SIZE - 1, i].set_xlabel(r'$\mathbf{z_{%d}}$' % i, fontsize=10)
    # axes[i, 0].set_ylabel(r'$\mathbf{z_{%d}}$' % i, fontsize=10)
    if dofig:
        plt.figure()


def plot_z_embeddings_from_distributions(enc, Zs_dists=None, dofig=True, stylename='z'):
    if Zs_dists is None:
        Zs_dists = generate_data(enc, [stylename], distributions=True)[stylename]

    num_points = Zs_dists.shape[0] // 10
    my_z_samples = np.concatenate([np.random.normal(loc=Zs_dists[:, :, 0], scale=Zs_dists[:, :, 1]) for _ in range(10)],
                                  axis=0)
    my_z_samples = my_z_samples[np.random.choice(np.arange(0, len(my_z_samples)), size=num_points, replace=False)]
    print(my_z_samples.shape)
    df = pd.DataFrame(my_z_samples, columns=[r'$\mathbf{stylename{%d}}$' % i for i in range(my_z_samples.shape[1])])
    print(df)
    p = seaborn.pairplot(df)
    # figs.suptitle(r'$Z \ Embeddings$', fontsize=30)
    # axes[Z_SIZE - 1, i].set_xlabel(r'$\mathbf{z_{%d}}$' % i, fontsize=10)
    # axes[i, 0].set_ylabel(r'$\mathbf{z_{%d}}$' % i, fontsize=10)
    if dofig:
        plt.figure()


def check_correlations(enc, Zs=None, stylename='z'):
    if Zs is None:
        Zs = generate_data(enc, [stylename])[stylename]
    num = Zs.shape[-1]
    for i in range(num):
        for j in range(i):
            if i != j:
                print(i, j, np.corrcoef(Zs[:, i], Zs[:, j])[0, 1])


def get_reconstruction_error(enc, dec, test_data=None, skip_obs_agents=False,shuffleu=False,ratedist=False,computecat=False):
    if test_data is None:
        batch_size = BATCH_SIZE
        _, test_data, _ = load_data.get_data(batch_size)

    if ratedist:
        _, errors,r,d = train.test(test_data, enc, dec, skip_obs_agents=skip_obs_agents,shuffleu=shuffleu,ratedist=True)
        return errors,r,d
    elif computecat:
        _,errors,labelcatlog,predcatlog = train.test(test_data, enc, dec, skip_obs_agents=skip_obs_agents, shuffleu=shuffleu,
                                     computecat=True)
        return errors,labelcatlog,predcatlog
    else:
        _, errors = train.test(test_data, enc, dec, skip_obs_agents=skip_obs_agents,shuffleu=shuffleu)
        return errors


def eval_single(savedir, modelname, training_data_name, load, epochs, checkpoint_path, results_save_dir=".",
                skip_obs_agents=False,justmeans=False):
    enc, dec, discrim, elbo_log, reconstruction_log, discrim_log = get_models(savedir, modelname, training_data_name,
                                                                              load, epochs, checkpoint_path,
                                                                              skip_obs_agents=skip_obs_agents)

    Zs, z_dists = generate_data(enc, ["z"], both=True)
    Zs = Zs["z"]
    z_dists = z_dists["z"]
    print("Minimum Standard Deviations:",z_dists[:, :, 1].min(axis=0))
    print("Maximum Standard Deviations:",z_dists[:, :, 1].max(axis=0))

    print("Minimum Mean:",z_dists[:, :, 0].min(axis=0))
    print("Maximum Mean:",z_dists[:, :, 0].max(axis=0))


    # check_correlations(enc, Zs)
    plot_z_embeddings(enc, Zs, dofig=False, title="Correlation between z embedding samples")
    plt.savefig(os.path.join(results_save_dir, "z_embeddings.png"),bbox_inches='tight')
    plt.clf()
    plot_z_embeddings(enc, z_dists[:, :, 0], dofig=False, title="Correlation between z embedding means")
    plt.savefig(os.path.join(results_save_dir, "z_embeddings_means.png"),bbox_inches='tight')
    plt.clf()

    if justmeans:
        return


    _, test_data, _ = load_data.get_data(batch_size=BATCH_SIZE)
    plot_reconstruction_error(enc, dec, test_data, skip_obs_agents=skip_obs_agents)
    plt.savefig(os.path.join(results_save_dir, "Reconstruction and Prediction Errors (Mean Euclidean Distance).png"),bbox_inches='tight')
    plt.clf()
    plotlogs(elbo_log, reconstruction_log, discrim_log)
    plt.savefig(os.path.join(results_save_dir, "training_logs2.png"),bbox_inches='tight')
    plt.clf()


def f1_loss(y_true,y_pred):

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    accuracy = (tp+tn)/(tp+tn+fp+fn)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1.item(),precision.item(),recall.item(),accuracy.item()

def multi_checkpoint_eval(results_dir, num, savedir="saved_models/checkpoints", load=True, skip_obs_agents=False):
    if not load:
        _, testdata, _ = get_data(499, 2, agent=True)
        elbos = []
        recons = []
        labelcatloglist = []
        predcatloglist = []
        for x in range(num):
            modelname = "checkpoint_" + str(x) + "_model.pkl"
            logname = modelname.replace("model", "logs")
            enc, dec, discrim, elbo_log, recon_log, discrim_log = get_models(savedir, modelname, logname, load=True,
                                                                             skip_obs_agents=skip_obs_agents)
            print("Model", x)
            e, r ,lc,pc = test(testdata, enc, dec, skip_obs_agents=skip_obs_agents,computecat=True)
            print("ELBO", np.mean(e))
            print("Reconstruction", [(k, np.mean(y)) for k, y in r.items()])
            elbos.append(e)
            recons.append(r)
            labelcatloglist.append(lc)
            predcatloglist.append(pc)
        # with open(os.path.join(results_dir, "test_data.pkl"), 'wb') as f:
        #     pickle.dump((elbos, recons,labelcatloglist,predcatloglist), f)
    else:
        with open(os.path.join(results_dir, "test_data.pkl"), 'rb') as f:
            modelname = "checkpoint_" + str(num - 1) + "_model.pkl"
            logname = modelname.replace("model", "logs")
            _, _, _, elbo_log, recon_log, discrim_log = get_models(savedir, modelname, logname, load=True,
                                                                   skip_obs_agents=skip_obs_agents)
            elbos, recons,labelcatloglist,predcatloglist = pickle.load(f)

    plotlogs(elbo_log, recon_log, discrim_log)
    plt.savefig(os.path.join(results_dir, "training_logs.png"),bbox_inches='tight')
    plt.clf()

    plot_with_std([{"ELBO": e} for e in elbos],10000, "Test Loss, Excluding 1% Extremes", outlier=0.01,y="Loss")
    plt.savefig(os.path.join(results_dir, "test_ELBO.png"),bbox_inches='tight')
    plt.clf()

    plot_with_std(recons, 10000, "Test Reconstruction Error", isrecon=True,y="Error")
    plt.savefig(os.path.join(results_dir, "test_reconstruction.png"),bbox_inches='tight')
    plt.clf()

    if not skip_obs_agents:
        plot_accuracies( labelcatloglist,predcatloglist,"Binarized Agent Observation Error",agents=True)
        plt.savefig(os.path.join(results_dir, "test_reconstruction_cat_agents.png"),bbox_inches='tight')
        plt.clf()

    plot_accuracies(labelcatloglist, predcatloglist, "Binarized Target Observation Error",agents=False)
    plt.savefig(os.path.join(results_dir, "test_reconstruction_cat_targets.png"),bbox_inches='tight')
    plt.clf()

def plot_accuracies(labelcatloglist,predcatloglist,title,agents):

    if agents:
        pre = "Agent"
        val = "obs_agents"
    else:
        pre = "Target"
        val = "obs_targets"


    tf1s = []
    taccs = []
    tpreds = []
    trecs = []
    for i in range(len(labelcatloglist)):
        f1, precision, recall, accuracy = f1_loss(labelcatloglist[i][val],predcatloglist[i][val])
        tf1s.append(f1)
        taccs.append(accuracy)
        tpreds.append(precision)
        trecs.append(recall)


    epochs = range(1,1+len(labelcatloglist))

    d = {
        "Epoch":epochs,
        pre+" F1 Score":tf1s,
        pre+" Accuracy":taccs,
        pre+" Precision":tpreds,
        pre+" Recall":trecs
    }

    df = pd.DataFrame(d)
    df = pd.melt(df, id_vars=["Epoch"])
    ax = seaborn.lineplot(x="Epoch", y="value", hue='variable', data=df)
    ax.set(title=title)
    ax.set(ylabel="Score")

def plot_with_std(values, maxnum, title,isrecon=False, outlier=0.0,y=None):
    if not isinstance(values, list):
        values = [values]
    l = len(list(values[0].values())[0])
    newlen = min(l, maxnum)
    outindx = int(outlier * newlen)
    pr = {}
    for i, r in enumerate(values):
        for k in r.keys():
            x = np.asarray(r[k])
            np.random.shuffle(x)
            v = x[0:newlen]
            if outindx > 0:
                v = np.sort(v)[outindx:-outindx]
            if k not in pr:
                pr[k] = v
            else:
                pr[k] = np.append(pr[k], v)

        if "Epoch" not in pr:
            pr['Epoch'] = np.ones(newlen-2*outindx) * (i + 1)
        else:
            pr["Epoch"] = np.append(pr['Epoch'], np.ones(newlen-2*outindx) * (i + 1))

    if isrecon:
        namedict = {
            'Epoch': 'Epoch',
            'pos': '"Positions' if 'knowledge' in pr.keys() else 'Position',
            'obs_targets': 'Target Obs.',
            'message': 'Message Reconstruction',
            'obs_agents': 'Agent Obs.',
            'targets': 'Target Positions',
            'messages':'Messages',
            'knowledge':'World Knowledge Reconstruction'}
        pr = {namedict[k]:v for k,v in pr.items() if k in list(namedict.keys())}

    df = pd.DataFrame(pr)
    df = pd.melt(df, id_vars=["Epoch"])
    ax = seaborn.lineplot(x="Epoch", y="value", hue='variable', data=df)
    ax.set(title=title)
    ax.set(ylabel=y)


def plot_reconstruction_error(enc, dec, data, dofig=True, skip_obs_agents=False, shuffleu=False):
    r = get_reconstruction_error(enc, dec, data, skip_obs_agents=skip_obs_agents,shuffleu=shuffleu)
    pr = {}
    for k2 in r.keys():
        r[k2] = np.asarray(r[k2])
        if k2 not in pr:
            pr[k2] = np.asarray(r[k2])
        else:
            pr[k2] = np.append(pr[k2], np.asarray(r[k2]))
    namedict = {
        'alpha': 'alpha',
        'pos': 'Position',
        'obs_targets': 'Target Obs.',
        'message': 'Message Reconstruction',
        'obs_agents': 'Agent Obs.'}

    pr2 = {}
    for k in pr.keys():
        pr2[namedict[k]] = pr[k]
    df = pd.DataFrame(pr2)

    ax = seaborn.boxplot(data=df, showfliers=False)
    ax.set(title=r'Reconstruction and Prediction Error (Mean Euclidean Distance)')
    ax.set(xlabel=None)

    if dofig:
        plt.figure()


def plot_many_reconstruction_error(encs, decs, data, values, valuname='alpha', titlesuffix=None, dofig=True, load=True):
    if not load:
        rs = [get_reconstruction_error(encs[i], decs[i], data) for i in range(len(encs))]
        with open("saved_many_reconstruction_data", "wb") as f:
            pickle.dump(rs, f)
    else:
        with open("saved_many_reconstruction_data", "rb") as f:
            rs = pickle.load(f)

    pr = {}
    for i, r in enumerate(rs):
        a = values[i]
        if valuname not in pr:
            pr[valuname] = np.ones(len(list(r.values())[0])) * a
        else:
            pr[valuname] = np.append(pr[valuname], np.ones(len(list(r.values())[0])) * a)
        for k2 in r.keys():
            r[k2] = np.asarray(r[k2])
            if k2 not in pr:
                pr[k2] = np.asarray(r[k2])
            else:
                pr[k2] = np.append(pr[k2], np.asarray(r[k2]))
    namedict = {
        'alpha': 'alpha',
        'pos': 'Position',
        'obs_targets': 'Target Obs.',
        'message': 'Message Reconstruction',
        'obs_agents': 'Agent Obs.'}

    pr2 = {}
    for k in pr.keys():
        pr2[namedict[k]] = pr[k]
    df = pd.DataFrame(pr2)
    df = pd.melt(df, id_vars="alpha", value_name="Error (MSE)")
    plt.rcParams['text.usetex'] = True
    ax = seaborn.boxplot(data=df, x="variable", y="Error (MSE)", hue=valuname, showfliers=False)
    ax.set(title=r'Reconstruction and Prediction Error (Mean Euclidean Distance) for Varying $\alpha$')
    ax.set(xlabel=None)
    if dofig:
        plt.figure()


def plotlogs(elbo_log, reconstruction_log, discrim_log, title='Training statistics',plotthree=False):
    t = range(len(elbo_log))
    fig = plt.figure(figsize=(6, 6))
    if plotthree:
        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        bl = fig.add_subplot(grid[1, 0])
        br = fig.add_subplot(grid[1, 1])
        bl.plot(t, reconstruction_log)
        bl.set_title("Mesage Reconstruction Distance")
        bl.get_xaxis().set_visible(False)
        br.plot(t, discrim_log)
        br.set_title("Discriminator Loss")
        fig.suptitle(title)

    else:
        grid = plt.GridSpec(1, 1, hspace=0.2, wspace=0.2)

    top = fig.add_subplot(grid[0, :])
    y = uniform_filter1d(elbo_log, size=1000)
    top.plot(t, elbo_log)
    top.plot(t, y)
    top.set_title("Model Loss")
    top.set_ylim([np.min(np.nan_to_num(elbo_log)), min(np.max(np.nan_to_num(elbo_log)), 100)])
    if not plotthree:
        top.set(xlabel="Batch")
        top.set(ylabel="Loss")





if __name__ == '__main__':
    # savedir = "saved_models"
    # modelname = "pickled_model_" + SAVE_NAME + ".pkl"
    # checkpoint_path = os.path.join("saved_models","checkpoints")
    # training_data_name = "training_data_" + SAVE_NAME + ".pkl"
    # load = True

    usecheckpointdict = {
        "partial": 9,
        "vanilla": 8,
        "nothing": 5,
        "noobs": 9, # small decoder
        "originalnoobs":8,
        "lbha":9,
        "nofactorvae": 6,
        "smalldecoder":5,
        "smalldecoder2":9,
        "lowbeta":8,
        "largedecodernoobs":9
    }

    domulti = False
    seaborn.set(font_scale=1.3)
    for type in ["noobs"]:#"vanilla","partial","nothing", "noobs", "nofactorvae"

        checkpointdir = "saved_models/" + type + "_checkpoints/"
        skip_obs_agents = False
        epochs = 10
        if type == "noobs" or type == "nothing" or type=="largedecodernoobs" or type =="smalldecodernoobs":
            skip_obs_agents = True
        if type == "vanilla":
            checkpointdir = "saved_models/checkpoints"

        print("Testing type ", type)
        if domulti:
            multi_checkpoint_eval("results/" + type, num=epochs, savedir=checkpointdir, load=False, skip_obs_agents=skip_obs_agents)
        else:
            modelname = "checkpoint_" + str(usecheckpointdict[type] - 1) + "_model.pkl"
            logsname = "checkpoint_" + str(usecheckpointdict[type] - 1) + "_logs.pkl"

            eval_single(checkpointdir, modelname, logsname, load=True, epochs=epochs, checkpoint_path=checkpointdir,
                        results_save_dir="results/" + type, skip_obs_agents=skip_obs_agents,justmeans=True)


