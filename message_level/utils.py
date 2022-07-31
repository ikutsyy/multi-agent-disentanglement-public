from numbers import Number

import numpy as np

import linecache
import os
import tracemalloc

from torch import softmax

from disentanglement.libs.probtorch import probtorch
from functools import wraps
import torch

from disentanglement.libs.probtorch.probtorch.objectives.montecarlo import log_like, kl
from disentanglement.message_level.model_parameters import CUDA


def np_sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig

def ifprint(a, *args):
    if a:
        print(*args)


def ifcuda(a):
    if CUDA:
        return a.cuda()
    return a

def get_rate_distortion(p,q,sample_dim=0, batch_dim=1,size_average=True, reduce=True):

    log_weights = q.log_joint(sample_dim, batch_dim, q.conditioned())
    d = -log_like(q, p, sample_dim, batch_dim, log_weights,
                     size_average=size_average, reduce=reduce).item()
    r = kl(q, p, sample_dim, batch_dim, log_weights,
                      size_average=size_average, reduce=reduce).item()

    return r,d



def permute_dims(zs):
    batch_size, dims = zs.shape
    newzs = torch.ones_like(zs)
    # Dimensions
    for i in range(dims):
        newvector = zs[:, i][torch.randperm(batch_size)]
        newzs[:, i] = newvector
    return newzs

def alltocuda(dict):
    for key in dict.keys():
        if isinstance(dict[key], torch.Tensor):
            dict[key] = ifcuda(dict[key])

def alldetatch(dict):
    for key in dict.keys():
        if isinstance(dict[key], torch.Tensor):
            dict[key] = dict[key].detach()


def alltocpu(dict):
    for key in dict.keys():
        if isinstance(dict[key], torch.Tensor):
            dict[key] = dict[key].cpu()


def print_trace(trace):
    for name, x in trace._nodes.items():
        if isinstance(x, torch.Tensor):
            print(name, x.shape)
        elif isinstance(x, probtorch.Stochastic):
            print(name, x.value.shape)
        else:
            print(name, type(x), x)


def expand_dicts(f):
    def expand_dict(d, num_samples):
        for k in d.keys():
            if hasattr(d[k], 'expand'):
                d[k] = d[k].expand(num_samples, *d[k].size())

    """Decorator that expands all input tensors to add a sample dimensions"""

    @wraps(f)
    def g(*args, **kwargs):
        num_samples = kwargs.get('num_samples', None)
        if num_samples is not None:
            for arg in args:
                if isinstance(arg, dict):
                    expand_dict(arg, num_samples)

            for k in kwargs:
                arg = kwargs[k]
                if isinstance(arg, dict):
                    expand_dict(arg, num_samples)
            return f(*args, **kwargs)
        else:
            return f(*args, **kwargs)

    return g

def np_logit(p):
    return np.log(p) - np.log(1 - p)

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))