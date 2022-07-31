import torch.nn as nn

from disentanglement.direct_model.direct_model_parameters import *


def send2cuda(x):
    if CUDA:
        return x.cuda()


class DirectPredictor(nn.Module):
    def __init__(self, num_agents, message_size, obs_size, size=DIRECT_MODEL_SCALE):
        super(self.__class__, self).__init__()
        self.num_agents = num_agents
        self.message_size = message_size
        self.obs_size = obs_size
        self.enc_hidden = nn.Sequential(
            nn.Linear(message_size, size * 2),
            nn.ReLU(),
            nn.Linear(size * 2, size * 2),
            nn.ReLU(),
            nn.Linear(size * 2, size),
            nn.ReLU(),
        )
        self.obs_targets_mean_std = nn.Sequential(
            nn.Linear(size, size * 2),
            nn.ReLU(),
            nn.Linear(size * 2, obs_size),
        )
        self.obs_agents_mean_std = nn.Sequential(
            nn.Linear(size, size * 2),
            nn.ReLU(),
            nn.Linear(size * 2, obs_size),
        )
        self.pos_mean_std = nn.Sequential(
            nn.Linear(size + 2 * obs_size, size * 2),
            nn.ReLU(),
            nn.Linear(size * 2, size),
            nn.ReLU(),
            nn.Linear(size, 2 ),
        )
        self.catcross = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    # We need num_samples for the parameter expansion to work correctly
    def forward(self, message,dosqrt = False):
        hidden = self.enc_hidden(message)
        obs_agents = self.obs_agents_mean_std(hidden)
        obs_targets = self.obs_targets_mean_std(hidden)

        pos = self.pos_mean_std(torch.cat([hidden,obs_agents, obs_targets], dim=-1))

        if not dosqrt:
            return {'pos': pos,
                    'obs_agents': obs_agents,
                    'obs_targets': obs_targets
                    }
        else:
            return {'pos': torch.sqrt(pos),
                    'obs_agents': torch.sqrt(obs_agents),
                    'obs_targets': torch.sqrt(obs_targets)
                    }

    def compute_loss(self, predicted, labels):
        loss = 0
        for k in predicted.keys():
            loss += self.mse(predicted[k], labels[k])
        return loss
