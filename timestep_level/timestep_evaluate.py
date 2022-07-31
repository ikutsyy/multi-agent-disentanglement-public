import os
import pickle

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter1d

from disentanglement.message_level import load_data
from disentanglement.message_level.evaluate import check_correlations, plot_z_embeddings, plot_with_std
from disentanglement.message_level.load_data import get_data
from disentanglement.timestep_level import timestep_train
from disentanglement.timestep_level.timestep_model_parameters import *
from disentanglement.timestep_level.timestep_train import get_models

def generate_data(enc, interested_values, test_data=None):
    if test_data is None:
        _, test_data, _ = get_data(BATCH_SIZE,agent=False)

    result = dict(zip(interested_values, [None] * len(interested_values)))
    for b, (knowledge, labels) in enumerate(test_data):
        if CUDA:
            knowledge = knowledge.cuda()
        q = enc(knowledge,BATCH_SIZE, num_samples=1)
        for value in interested_values:
            z = q[value].value.squeeze().cpu().detach().numpy()
            if result[value] is None:
                result[value] = z
            else:
                result[value] = np.append(result[value], z, axis=0)
    return result

def eval_single(*args):
    enc, dec, discrim, elbo_log, reconstruction_log, discrim_log = get_models(*args)

    Zs = generate_data(enc, ["agent_behavior"])["agent_behavior"]
    Zs = Zs.reshape((Zs.shape[0],-1))
    print(Zs.shape)
    check_correlations(enc, Zs,stylename="agent_behavior")
    plot_z_embeddings(enc, Zs,stylename="agent_behavior")

    plotlogs(elbo_log, reconstruction_log, discrim_log)
    plt.show()


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

def get_reconstruction_error(enc, dec, test_data=None,computecat=False):
    if test_data is None:
        batch_size = BATCH_SIZE
        _, test_data, _ = load_data.get_data(batch_size,agent=False)

    elif computecat:
        _,errors,labelcatlog,predcatlog = timestep_train.test(test_data, enc, dec,
                                     computecat=True)
        return errors,labelcatlog,predcatlog
    else:
        _, errors = timestep_train.test(test_data, enc, dec)
        return errors

def collectandplotelbo(resultsdir,num, savedir):
    seaborn.set(font_scale=1.3)
    x = num-1
    modelname = "checkpoint_" + str(x) + "_model.pkl"
    logname = modelname.replace("model", "logs").replace("checkpoint_","checkpoint")
    _, _, _, elbo_log, _, _ = get_models(savedir, modelname, logname, load=True)
    plotlogs(elbo_log,None,None)
    plt.savefig(os.path.join(resultsdir, "training_log_simple.png"),bbox_inches='tight')
    plt.clf()



def multi_checkpoint_eval(results_dir,num,savedir="saved_models/checkpoints",load=True):
    if not load:
        _, testdata, _ = get_data(499 * 2, 2, agent=False)
        elbos = []
        recons = []
        for x in range(num):
            modelname = "checkpoint_" + str(x) + "_model.pkl"
            logname = modelname.replace("model", "logs").replace("checkpoint_","checkpoint")
            enc, dec, discrim, elbo_log, recon_log, discrim_log = get_models(savedir, modelname, logname, load=True)
            print("Model", x)
            e, r = timestep_train.test(testdata, enc, dec)
            print("ELBO", np.mean(e))
            print("Reconstruction", [(k, np.mean(y)) for k, y in r.items()])
            elbos.append(e)
            recons.append(r)
        with open(os.path.join(results_dir,"test_data.pkl"), 'wb') as f:
            pickle.dump((elbos, recons), f)
    else:
        with open(os.path.join(results_dir,"test_data.pkl"), 'rb') as f:
            modelname = "checkpoint_" + str(num-1) + "_model.pkl"
            logname = modelname.replace("model", "logs")
            _, _, _, elbo_log, recon_log, discrim_log = get_models(savedir, modelname, logname, load=True)
            elbos, recons = pickle.load(f)

    plotlogs(elbo_log, recon_log, discrim_log)
    plt.savefig(os.path.join(results_dir, "training_logs.png"), bbox_inches='tight')
    plt.clf()

    plot_with_std([{"ELBO": e} for e in elbos], 10000, "Test Loss, Excluding 1% Extremes", outlier=0.01, y="Loss")
    plt.savefig(os.path.join(results_dir, "test_ELBO.png"), bbox_inches='tight')
    plt.clf()

    plot_with_std(recons, 10000, "Test Reconstruction Error", isrecon=True, y="Error")
    plt.savefig(os.path.join(results_dir, "test_reconstruction.png"), bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    savedir = "saved_models"
    modelname = "pickled_model_" + SAVE_NAME + ".pkl"
    checkpoint_path = os.path.join("saved_models", "checkpoints")
    training_data_name = "training_data_" + SAVE_NAME + ".pkl"
    load = False
    epochs = 10
    # eval_single(savedir, modelname, training_data_name, load, epochs, checkpoint_path)
    collectandplotelbo("results",12,"saved_models/checkpoints")
    # multi_checkpoint_eval(results_dir="results",num=12,load=False)