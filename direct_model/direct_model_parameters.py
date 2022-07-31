import multiprocessing

import torch

MULTIPROCESSING_METHOD = 'fork'
# CUDA
CUDA = torch.cuda.is_available()
if CUDA and multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method(MULTIPROCESSING_METHOD)


MESSAGE_SIZE=64+2

DIRECT_MODEL_SCALE = 128

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPS = 1e-9

NUM_WORKERS = 2