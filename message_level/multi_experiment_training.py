import os
import pickle
import sys

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
import torch
from disentanglement.message_level import model_parameters, train, load_data
from disentanglement.message_level.evaluate import generate_data, check_correlations, plot_z_embeddings, plotlogs, \
    get_reconstruction_error, plot_z_embeddings_from_distributions, plot_reconstruction_error, \
    plot_many_reconstruction_error

from disentanglement.message_level.model_parameters import *
import pprint

from disentanglement.message_level.train import get_models


def doplots(blankname,savedir, enc, dec, test_data,elbo_log,reconstruction_log,discrim_log):
    plot_reconstruction_error(enc, dec, test_data, dofig=False)
    plt.savefig(os.path.join(savedir, "reconstruction_errors_" + blankname+ ".png"),bbox_inches='tight')
    plt.clf()
    z_samples, z_dist = [x["z"] for x in generate_data(enc, ["z"], test_data=test_data, both=True)]
    check_correlations(enc, z_samples)
    plot_z_embeddings(enc, z_samples, dofig=False)
    plt.savefig(os.path.join(savedir, "z_embeddings_" + blankname + ".png"),bbox_inches='tight')
    plt.clf()
    plot_z_embeddings(enc, z_dist[:, :, 0], dofig=False)
    plt.savefig(os.path.join(savedir, "z_embeddings_means_" + blankname + ".png"),bbox_inches='tight')
    plt.clf()

    if isinstance(reconstruction_log[0], torch.Tensor):
        reconstruction_log = np.ones(len(reconstruction_log))

    plotlogs(elbo_log, reconstruction_log, discrim_log,
             title='Training statistics for run ' + blankname)

    plt.savefig(os.path.join(savedir, "trainlog_" + blankname + ".png"),bbox_inches='tight')
    plt.figure()


def train_cube(savedir, epochs, alphas=None, betas=None,
                      gammas=None,z_sizes=None):
    def dump(model_path,training_data_path,enc,dec,discriminator,elbo_log,reconstruction_log,discrim_log):
        with open(model_path, "wb") as f:
            pickle.dump((enc, dec,discriminator), f)
        with open(training_data_path, "wb") as f:
            pickle.dump((elbo_log, reconstruction_log,discrim_log), f)
    def load(model_path,training_data_path):
        with open(model_path, "rb") as f:
            enc, dec,discriminator = pickle.load(f)
        with open(training_data_path, "rb") as f:
            elbo_log, reconstruction_log, discrim_log = pickle.load(f)
        return enc, dec,discriminator,elbo_log, reconstruction_log, discrim_log

    if gammas is None:
        gammas = [weights["G"]]
    if betas is None:
        betas = [weights["B"]]
    if alphas is None:
        alphas = [weights["A"]]
    if z_sizes is None:
        z_sizes = [model_parameters.Z_SIZE]

    _, test_data, _ = load_data.get_data(batch_size=BATCH_SIZE)


    for z_size in z_sizes:
        for alpha in alphas:
            for beta in betas:
                for gamma in gammas:
                    weights["B"] = beta
                    weights["A"] = alpha
                    weights["G"] = gamma
                    modelname = "pickled_model_" + SAVE_NAME + "_z_" + str(z_size) +"_a_"+str(alpha)+"_b_"+str(beta)+"_g_"+str(gamma)+ ".pkl"
                    trainname = modelname.replace("pickled_model","training_data")
                    training_data_path = os.path.join(savedir,trainname)
                    model_path = os.path.join(savedir, modelname)
                    print(model_path)
                    if os.path.exists(model_path):
                        print("loading from file")
                        enc, dec, discriminator, elbo_log, reconstruction_log, discrim_log = load(model_path,training_data_path)
                    else:
                        print("training...")
                        enc, dec,discriminator, elbo_log, reconstruction_log,discrim_log = train.train_model(epochs, sample_size=NUM_SAMPLES,do_test=False)
                        dump(model_path,training_data_path,enc,dec,discriminator, elbo_log,reconstruction_log,discrim_log)

                    blankname = modelname.replace("pickled_model_",'').replace(".pkl","")
                    doplots(blankname,os.path.join(savedir, "results"),enc,dec,test_data,elbo_log,reconstruction_log,discrim_log)




def evaluate_all_saved_models(dir, excluded=None, kdict=None):

    if kdict is None:
        kdict = {"z": "Z Values", "a": 'Alpha', "s": "Sample Size", "ba": "Batch Size", "g":"Gamma"}
    if excluded is None:
        excluded = []
    plt.rcParams['text.usetex'] = True
    files = os.listdir(dir)
    savedir = os.path.join(dir,"results")
    savedir_files = os.listdir(savedir)

    _, test_data, _ = load_data.get_data(batch_size=BATCH_SIZE)
    for x in files:
        if x.endswith(".pkl") and x.startswith("pickled_model") and x not in excluded:
            cut = x.split("_")
            if len(cut) == 5:
                name = cut[2]
                k = cut[3]
                param = cut[4][:-4]
                original_z = model_parameters.Z_SIZE
                if k == "z":
                    model_parameters.Z_SIZE = int(param)
                train_name = "training_data_"+name+"_"+k+"_"+param+".pkl"

                if "trainlog"+name+"_"+k+"_"+param+".png" not in savedir_files:

                    enc, dec,_, elbo_log, reconstruction_log, discrim_log = get_models(dir,x,train_name,load=True)
                    print(x)
                    plot_reconstruction_error(enc,dec,test_data,dofig=False)
                    plt.savefig(os.path.join(savedir, "reconstruction_errors" + name + "_" + k + "_" + param + ".png"),bbox_inches='tight')
                    plt.clf()
                    z_samples,z_dist = [x["z"] for x in generate_data(enc, ["z"],test_data=test_data,both=True)]
                    check_correlations(enc, z_samples)
                    plot_z_embeddings(enc, z_samples,dofig=False)
                    plt.savefig(os.path.join(savedir, "z_embeddings_" + name + "_" + k + "_" + param + ".png"),bbox_inches='tight')
                    plt.clf()
                    plot_z_embeddings(enc, z_dist[:,:,0],dofig=False)
                    plt.savefig(os.path.join(savedir, "z_embeddings_means_" + name + "_" + k + "_" + param + ".png"),bbox_inches='tight')
                    plt.clf()

                    if isinstance(reconstruction_log[0],torch.Tensor):
                        reconstruction_log = np.ones(len(reconstruction_log))
                    plotlogs(elbo_log, reconstruction_log, discrim_log,title='Training statistics for run '+kdict[k]+" = "+param)

                    plt.savefig(os.path.join(savedir,"trainlog"+name+"_"+k+"_"+param+".png"),bbox_inches='tight')
                    plt.figure()
                model_parameters.Z_SIZE = original_z

if __name__ == '__main__':
   z_sizes = [4]
   alphas = [0.001, 0.1, 1, 10, 100]
   betas = [1]
   gammas = [0, 1, 10]
   savedir = "test_models"

   encs = []
   decs = []
   for alpha in alphas:
        modelname = "pickled_model_large_z_4_a_" + str(alpha) + "_b_1_g_0.pkl"
        with open(os.path.join(savedir, modelname), "rb") as f:
            enc, dec, _ = pickle.load(f)
            encs.append(enc)
            decs.append(dec)
   _, test_data, _ = load_data.get_data(batch_size=499)
   plot_many_reconstruction_error(encs, decs, test_data, alphas, titlesuffix="", dofig=False, load=True)
   plt.savefig("varying_alpha.png",bbox_inches='tight')
   #
   # for gamma in gammas:
   #     for alpha in alphas:
   #         modelname = "pickled_model_large_z_4_a_" + str(alpha) + "_b_1_g_"+str(gamma)+".pkl"
   #         with open(os.path.join(savedir, modelname), "rb") as f:
   #             enc, dec, _ = pickle.load(f)
   #             encs.append(enc)
   #             decs.append(dec)
   #
   #     _, test_data, _ = load_data.get_data(batch_size=499)
   #
   #     plot_many_reconstruction_error(encs, decs, test_data, alphas,titlesuffix="", dofig=False, load=True)
   #     plt.savefig("./varying_alpha_b_1_g_"+str(gamma)+".png")
   #     # plt.show()


# if __name__ == '__main__':
#      savedir = "test_models"
#      modelname = "pickled_model_" + SAVE_NAME + ".pkl"
#      checkpoint_path = os.path.join(savedir, "checkpoints")
#      training_data_name = "training_data_" + SAVE_NAME + ".pkl"
#      epochs = 5
#      z_sizes = [4]
#      if len(sys.argv)>1:
#          alphas = [float(sys.argv[1])]
#          betas = [float(sys.argv[2])]
#          gammas = [float(sys.argv[3])]
#      else:
#          alphas = [0.001,0.1,1,10,100]
#          betas = [1]
#          gammas = [0,1,10]
#      print(alphas,betas,gammas)
#      train_cube(savedir,epochs,alphas,betas,gammas,z_sizes)
#      evaluate_all_saved_models("saved_models")
