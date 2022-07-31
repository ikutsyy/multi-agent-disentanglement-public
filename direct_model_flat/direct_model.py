import torch.nn as nn

from disentanglement.direct_model_flat.direct_model_parameters import *


def send2cuda(x):
    if CUDA:
        return x.cuda()


class DirectPredictor(nn.Module):
    def __init__(self, num_agents, message_size, obs_size, size=DIRECT_MODEL_SCALE):
        super(self.__class__, self).__init__()
        self.num_agents = num_agents
        self.message_size = message_size
        self.obs_size = obs_size
        self.network = nn.Sequential(
            nn.Linear(message_size, 8 * size),
            nn.ReLU(),
            nn.Linear(8 * size, 4 * size),
            nn.ReLU(),
            nn.Linear(4 * size, 2 * size),
            nn.ReLU(),
            nn.Linear(2 * size, 1 * size),
            nn.ReLU(),
            nn.Linear(size, 2 + obs_size + obs_size),
        )
        self.catcross = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    # We need num_samples for the parameter expansion to work correctly
    def forward(self, message):
        predicted = self.network(message)
        return {'pos': predicted[:, 0:2],
                'obs_agents': predicted[:, 2: 2 + self.obs_size],
                'obs_targets': predicted[:,2 + self.obs_size:]
                }

    def compute_loss(self, predicted, labels):
        loss = 0
        for k in predicted.keys():
            loss += self.mse(predicted[k], labels[k])
        return loss
