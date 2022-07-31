import multiprocessing

import torch




MULTIPROCESSING_METHOD = 'fork'
# CUDA
CUDA = torch.cuda.is_available()
if CUDA and multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method(MULTIPROCESSING_METHOD)



MESSAGE_SIZE=64+2
KNOWLEDGE_SIZE=128

# Model Paramters:

DISCRIMINATOR_SIZE = 64

_DEFAULT_SIZE = 128

Z_SIZE = 4
ENCODER_HIDDEN1_SIZE = _DEFAULT_SIZE
ENCODER_HIDDEN2_SIZE = _DEFAULT_SIZE
ENCODER_HIDDEN3_SIZE = _DEFAULT_SIZE
ENCODER_HIDDEN4_SIZE = _DEFAULT_SIZE

DECODER_HIDDEN1_SIZE = _DEFAULT_SIZE
DECODER_HIDDEN2_SIZE = _DEFAULT_SIZE
DECODER_HIDDEN3_SIZE = _DEFAULT_SIZE
DECODER_HIDDEN4_SIZE = _DEFAULT_SIZE

# Processor parameters
PREPROCESS_CORES = 2

# Data parameters
SAVE_NAME = "large"
NUM_WORKERS = 2


DATASET_NAME = SAVE_NAME+"_dataset.pkl"
PARAMS_FILE = "../../data/params.json"
MULTI_FILE_DIRECTORY = "../../data/"+SAVE_NAME+"_multi"

CHUNK_SIZE_MAP = {
    "large":100*499,
    "medium":100*499,
    "small":100*499,
    "val":100*499,
    "partial":100*499,
    "small_val":100*499,
    "adversarial":100*499
}

TRAIN_CUTOFF = 24950000000000000
TRAIN_SAVE_BATCHES = 20000

DEBUG = False

# Training Parameters:
DATA_PATH = "../../data"
PREFETCH = 2
PERSISTENT_WORKERS = True
PIN_MEMORY = True

BATCH_SIZE = 64
LABEL_FRACTION = 1.0
LEARNING_RATE = 1e-4
EPS = 1e-9

NUM_SAMPLES = 8

# LOSS parameters:

# Higher beta - worse reconstruction, more normal latent
# Higher alpha - higher supervised reconstruction quality
weights = {"A":1,"B":1,"G":1}

PREPROCESS_OBS = True

