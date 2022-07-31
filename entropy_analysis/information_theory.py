import math
import os
import pickle
import sys

import numpy
import torch

from disentanglement.message_level import train as message_level_train, utils
from disentanglement.message_level import load_data
from disentanglement.message_level.utils import np_logit

import numpy as np


# Adapted from scipy.differential_entropy
def pad_along_last_axis(X, m):
    """Pad the data for computing the rolling window difference."""
    shape = np.array(X.shape)
    shape[-1] = m
    Xl = np.broadcast_to(X[..., [0]], shape)  # [0] vs 0 to maintain shape
    Xr = np.broadcast_to(X[..., [-1]], shape)
    return np.concatenate((Xl, X, Xr), axis=-1)


# Adapted from scipy.differential_entropy
def differential_entropy(X):
    """Compute the Vasicek estimator"""
    X = np.sort(X, axis=-1)
    n = X.shape[-1]
    m = math.floor(math.sqrt(n) + 0.5)
    X = pad_along_last_axis(X, m)
    differences = X[..., 2 * m:] - X[..., : -2 * m:]
    differences = np.abs(np.extract(differences != 0, differences))

    logs = np.log(n / (2 * m) * differences)
    return np.mean(logs, axis=-1)


def load_data_array(data_dir, num_datasets, type):
    data = [np.load(os.path.join(data_dir, x))[type]
            for i, x in enumerate(os.listdir(data_dir))
            if "agent" in x]

    data = np.concatenate(data[:min(len(data), num_datasets)])
    return data


def load_from_pickles(data_dir, type):
    all = []
    for i, x in enumerate(os.listdir(data_dir)):
        print(x)
        with open(os.path.join(data_dir, x), 'rb') as pk:
            df = pickle.load(pk)
            arr = np.asarray(df[type])
            all.append(arr)

    return np.asarray(all).reshape(-1)


def compute_rate(data, enc,
                 mmu=None,
                 mstd=None):

    if mstd is None:
        mstd = [34.02126, 44.81286, 30.820868, 34.11791]
    if mmu is None:
        mmu = [-4.73177, -1.4114183, 1.48519, 0.14848284]
    marginal = torch.distributions.normal.Normal(loc=torch.tensor(mmu), scale=torch.tensor(mstd))
    klds = []
    for x, _ in data:
        # Computed previously
        q,dists = enc(x.cuda(), num_samples=1, extract_distributions=True)
        p_z_x = dists['z'].cpu()
        p_z_x = torch.distributions.normal.Normal(loc=p_z_x[:,:,0],
                                                  scale=p_z_x[:,:,1])
        kld = torch.distributions.kl_divergence(p_z_x,marginal)
        klds.append(kld.detach().numpy())
        break
    expected_kld = numpy.concatenate(klds).mean(axis=0)
    print(expected_kld)



def compute_z_marginal(data, enc):
    mus = []
    stds = []
    for x, _ in data:
        x = utils.ifcuda(x)
        q, distributions = enc(x, num_samples=1, extract_distributions=True)
        mus.append(distributions['z'][:, :, 0].detach().cpu().numpy())
        stds.append(distributions['z'][:, :, 1].detach().cpu().numpy())

    mus = np.concatenate(mus)
    stds = np.concatenate(stds)
    mmu = mus.mean(axis=0)
    mstd = np.sqrt(np.square(stds).mean(axis=0))
    print("Mu:", mmu)
    print("Std:", mstd)
    return mmu,mstd



# For checkpoint 2, large
# Mu: [-4.73177    -1.4114183   1.48519     0.14848284]
# Std: [34.172787 44.272926 31.090162 34.386017]

def compute_ranges(data):
    print(np.ptp(data, axis=0))


def compute_entropy(data):
    entropies = np.asarray([differential_entropy(data[:, i]) for i in range(data.shape[-1])])
    for i in range(data.shape[-1]):
        print(i, "has entropy", entropies[i])
    mean = entropies.mean()
    print(mean)
    print("entropies:", entropies)
    mask = [1 if x > mean else 0 for x in entropies]
    print("mask:", mask)

def num_unique_episodes():
    eps = load_from_pickles("../../data/large_multi", "episode")
    numepisodes = len(os.listdir("../../data/large_multi")) * 100
    print(eps.shape)
    nunique = len(np.unique(eps))
    print(nunique, "Unique episodes, so only ", 100 * (1 - (nunique) / numepisodes), "% are repeats")


def get_data(agent=True):
    _, test_data, _ = load_data.get_data(499, agent=agent)
    return test_data


def get_model(agent=True):
    savedir = "../message_level/saved_models"
    modelname = "nofactorvae_pickled_model_large.pkl"
    training_data_name = "nofactorvae_training_data_large.pkl"

    if agent:
        enc, dec, _, _, _, _ = message_level_train.get_models(savedir, modelname, training_data_name, load=True)
        return enc, dec


if __name__ == '__main__':
    data = get_data()
    enc, dec = get_model()
    mmu,mstd = compute_z_marginal(data,enc)
    compute_rate(data, enc,mmu,mstd)
    # num_unique_episodes()
    # data = load_data_array("../../data/large/train", 24,"message")
    # compute_ranges(data)
