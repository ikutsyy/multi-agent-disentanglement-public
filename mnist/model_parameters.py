import torch

# Model Paramters:
NUM_PIXELS = 784
NUM_HIDDEN1 = 400
NUM_HIDDEN2 = 200
NUM_STYLE = 10
NUM_DIGITS = 10

# Training Parameters:
NUM_SAMPLES = 2
NUM_BATCH = 128
LABEL_FRACTION = 0.1
LEARNING_RATE = 1e-3
EPS = 1e-9
BIAS_TRAIN = (60000 - 1) / (NUM_BATCH - 1)
BIAS_TEST = (10000 - 1) / (NUM_BATCH - 1)
CUDA = torch.cuda.is_available()

# LOSS parameters:

weights = {}
weights["A"] = 0.1
weights["B"] = (4.0, 1.0, 1.0, 0.0, 1.0)

DATA_PATH = 'datasets'
