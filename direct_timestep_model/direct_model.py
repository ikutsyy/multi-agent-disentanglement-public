import torch
import torch.nn as nn

from disentanglement.direct_timestep_model.direct_model_parameters import *


def send2cuda(x):
    if CUDA:
        return x.cuda()


class DirectPredictor(nn.Module):
    def __init__(self, num_agents, num_targets, message_size, obs_size, size=DIRECT_MODEL_SCALE):
        super(self.__class__, self).__init__()
        self.num_agents = num_agents
        self.message_size = message_size
        self.obs_size = obs_size
        self.num_targets = num_targets
        self.network = nn.Sequential(
            nn.Linear(KNOWLEDGE_SIZE, 8 * size),
            nn.ReLU(),
            nn.Linear(8 * size, 4 * size),
            nn.ReLU(),
            nn.Linear(4 * size, 2 * size),
            nn.ReLU(),
            nn.Linear(2 * size, 1 * size),
            nn.ReLU(),
            nn.Linear(size, 2*num_agents + 2*num_targets + num_agents*obs_size*2 + num_agents*message_size),
        )
        self.catcross = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    # We need num_samples for the parameter expansion to work correctly
    def forward(self, message,dosqrt=False):
        predicted = self.network(message)
        batch_size = message.shape[0]
        possize = self.num_agents*2
        tarsize = self.num_targets*2
        obsize = self.num_agents*self.obs_size
        messize = self.num_agents*self.message_size
        if not dosqrt:
            return {'positions': torch.reshape(predicted[:, 0:possize],
                                               (batch_size,self.num_agents,2)),
                    'targets': torch.reshape(predicted[:, possize:possize+tarsize],
                                               (batch_size, self.num_targets, 2)),
                    'obs_agents':torch.reshape(predicted[:,possize+tarsize:possize+tarsize+obsize],
                                               (batch_size,self.num_agents,self.obs_size)),
                    'obs_targets': torch.reshape(predicted[:, possize + tarsize+obsize:possize + tarsize + 2*obsize],
                                                (batch_size, self.num_agents, self.obs_size)),
                    'messages': torch.reshape(predicted[:,possize + tarsize+2*obsize:],
                                              (batch_size,self.num_agents,self.message_size))
                    }
        else:
            return {'positions': torch.sqrt(torch.reshape(predicted[:, 0:possize],
                                               (batch_size, self.num_agents, 2))),
                    'targets': torch.sqrt(torch.reshape(predicted[:, possize:possize + tarsize],
                                             (batch_size, self.num_targets, 2))),
                    'obs_agents': torch.sqrt(torch.reshape(predicted[:, possize + tarsize:possize + tarsize + obsize],
                                                (batch_size, self.num_agents, self.obs_size))),
                    'obs_targets': torch.sqrt(torch.reshape(
                        predicted[:, possize + tarsize + obsize:possize + tarsize + 2 * obsize],
                        (batch_size, self.num_agents, self.obs_size))),
                    'messages': torch.sqrt(torch.reshape(predicted[:, possize + tarsize + 2 * obsize:],
                                              (batch_size, self.num_agents, self.message_size)))
                    }

    def compute_loss(self, predicted, labels):
        loss = 0
        for k in predicted.keys():
            loss += self.mse(predicted[k], labels[k])
        return loss
