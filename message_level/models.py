import torch
from torch.nn.functional import cross_entropy, relu, softplus

import disentanglement
from disentanglement.libs.probtorch import probtorch
import torch.nn as nn
from disentanglement.libs.probtorch.probtorch.util import expand_inputs

from disentanglement.message_level.model_parameters import *
from disentanglement.message_level.utils import expand_dicts, ifcuda


def send2cuda(x):
    if CUDA:
        return x.cuda()


class Disentnanglement_Discriminator(nn.Module):
    def __init__(self, z_size, hidden_size=DISCRIMINATOR_SIZE):
        super(self.__class__, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(z_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, z):
        return self.network(z)


class Encoder(nn.Module):
    def __init__(self, num_agents, message_size, obs_size, z_size=Z_SIZE, hidden_message_size=ENCODER_HIDDEN1_SIZE,
                 hidden2_size=ENCODER_HIDDEN2_SIZE, hidden3_size=ENCODER_HIDDEN3_SIZE,
                 hidden4_size=ENCODER_HIDDEN4_SIZE,skip_obs_agents=False):
        super(self.__class__, self).__init__()
        num_obs_portions = 1 if skip_obs_agents else 2
        self.skip_obs_agents = skip_obs_agents
        self.enc_hidden = nn.Sequential(
            nn.Linear(message_size, hidden_message_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_message_size * 2, hidden_message_size*2),
            nn.ReLU(),
            nn.Linear(hidden_message_size * 2, hidden_message_size),
            nn.ReLU(),
        )
        self.obs_targets_mean_std = nn.Sequential(
            nn.Linear(hidden_message_size, hidden_message_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_message_size * 2, obs_size * 2),
        )
        self.obs_agents_mean_std = nn.Sequential(
            nn.Linear(hidden_message_size, hidden_message_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_message_size * 2, obs_size * 2),
        )
        self.pos_mean_std = nn.Sequential(
            nn.Linear(hidden_message_size + num_obs_portions * obs_size, hidden3_size * 2),
            nn.ReLU(),
            nn.Linear(hidden3_size * 2, hidden3_size),
            nn.ReLU(),
            nn.Linear(hidden3_size, 2 * 2),
        )
        self.z_mean_std = nn.Sequential(
            nn.Linear(hidden_message_size + num_obs_portions * obs_size + 2, hidden4_size * 2),
            nn.ReLU(),
            nn.Linear(hidden4_size * 2, hidden4_size),
            nn.ReLU(),
            nn.Linear(hidden4_size, z_size * 2)
        )
        self.agent_temp = 0.66

    @expand_inputs
    @expand_dicts
    # We need num_samples for the parameter expansion to work correctly
    def forward(self, messages, labels=None, num_samples=NUM_SAMPLES, extract_distributions=False):
        if labels is None:
            labels = {'pos': None, 'obs_agents': None,
                      'obs_targets': None, 'z': None}

        q = probtorch.Trace()

        def combined_normal(name, weights):
            if len(weights.shape) == 2:
                return q.normal(loc=weights[:, 0:weights.shape[-1] // 2],
                                scale=softplus(weights[:, weights.shape[-1] // 2:]) + EPS,
                                value=labels[name],
                                name=name)
            else:
                return q.normal(loc=weights[:, :, 0:weights.shape[-1] // 2],
                                scale=softplus(weights[:, :, weights.shape[-1] // 2:])+ EPS,
                                value=labels[name],
                                name=name)

        message_hidden = torch.nan_to_num(self.enc_hidden(messages))
        if not self.skip_obs_agents:
            obs_agents_weights = self.obs_agents_mean_std(message_hidden)
            obs_agents = combined_normal('obs_agents', obs_agents_weights)

        obs_targets_weights = self.obs_targets_mean_std(message_hidden)
        obs_targets = combined_normal('obs_targets', obs_targets_weights)

        if not self.skip_obs_agents:
            pos_weights = self.pos_mean_std(torch.cat([message_hidden, obs_agents, obs_targets], dim=-1))
        else:
            pos_weights = self.pos_mean_std(torch.cat([message_hidden, obs_targets], dim=-1))

        pos = combined_normal('pos', pos_weights)

        if not self.skip_obs_agents:
            z_weights = self.z_mean_std(torch.cat([
                message_hidden, pos, obs_agents, obs_targets
            ], dim=-1))
        else:
            z_weights = self.z_mean_std(torch.cat([
                message_hidden, pos, obs_targets
            ], dim=-1))

        if len(pos_weights.shape) == 2:
            z = q.normal(loc=z_weights[:, 0:z_weights.shape[-1] // 2],
                         scale=softplus(z_weights[:, z_weights.shape[-1] // 2:])+ EPS,
                         name='z')
        else:
            z = q.normal(loc=z_weights[:, :, 0:z_weights.shape[-1] // 2],
                         scale=softplus(z_weights[:, :, z_weights.shape[-1] // 2:]) + EPS,
                         name='z')

        if extract_distributions:
            dist_map = {
                "pos": pos_weights,
                "obs_targets": obs_targets_weights,
                "z": z_weights
            }
            if not self.skip_obs_agents:
                dist_map["obs_agents"] = obs_agents_weights
            for k in dist_map.keys():
                # Separate out mean and std
                dist_map[k] = torch.squeeze(torch.stack([dist_map[k][:,:, 0:dist_map[k].shape[-1] // 2],
                                           softplus(dist_map[k][:, :, dist_map[k].shape[-1] // 2:]) + EPS],dim=-1),dim=0)
            return q,dist_map


        return q


def binary_cross_entropy(x_mean, x, EPS=1e-9):
    return - (torch.log(x_mean + EPS) * x +
              torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1)


class Decoder(nn.Module):
    def __init__(self, num_agents, message_size, obs_size, z_size=Z_SIZE, pos_hidden_size=DECODER_HIDDEN1_SIZE,
                 z_hidden_size=DECODER_HIDDEN2_SIZE, obs_hidden_size=DECODER_HIDDEN3_SIZE,
                 message_hidden_size=DECODER_HIDDEN4_SIZE,skip_obs_agents=False):
        super(self.__class__, self).__init__()

        num_obs_portions = 1 if skip_obs_agents else 2
        self.skip_obs_agents = skip_obs_agents

        self.sizes = {'pos': 2,
                      'obs_agents': obs_size,
                      'obs_targets': obs_size,
                      'z': z_size
                      }

        self.pos_hidden = nn.Sequential(
            nn.Linear(2, pos_hidden_size * 2),
            nn.ReLU(),
            nn.Linear(pos_hidden_size * 2, pos_hidden_size),
            nn.ReLU()
        )
        self.obs_agents_mean_std = nn.Sequential(
            nn.Linear(pos_hidden_size, obs_hidden_size * 2),
            nn.ReLU(),
            nn.Linear(obs_hidden_size * 2, 2 * obs_size)

        )
        self.obs_targets_mean_std = nn.Sequential(
            nn.Linear(pos_hidden_size, obs_hidden_size * 2),
            nn.ReLU(),
            nn.Linear(obs_hidden_size * 2, 2 * obs_size)
        )

        self.z_mean_std = nn.Sequential(
            nn.Linear(pos_hidden_size + num_obs_portions * obs_size, z_hidden_size * 2),
            nn.ReLU(),
            nn.Linear(z_hidden_size * 2, z_hidden_size),
            nn.ReLU(),
            nn.Linear(z_hidden_size, 2 * z_size)
        )

        self.message_network = nn.Sequential(
            nn.Linear(num_obs_portions * obs_size + 2 + z_size, message_hidden_size * 2),
            nn.ReLU(),
            nn.Linear(message_hidden_size * 2, message_hidden_size),
            nn.ReLU(),
            nn.Linear(message_hidden_size, message_size),
            nn.Sigmoid()
        )
        #
        # self.num_style = num_style
        # self.num_digits = num_digits
        self.agent_temp = 0.66
        # self.dec_images = nn.Sequential(
        #     nn.Linear(num_hidden1, num_pixels),
        #     nn.Sigmoid())

    def forward(self, messages, q=None, batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES,shuffledu=None):
        p = probtorch.Trace()

        if q is None:
            q = {'agent': None, 'pos': None, 'obs_agents': None,
                 'obs_targets': None, 'z': None}

        def combined_normal(name, weights, value=None):
            if value is None:
                value=q[name]
            if len(weights.shape) == 2:
                return p.normal(loc=weights[:, 0:weights.shape[-1] // 2],
                                scale=softplus(weights[:, weights.shape[-1] // 2:]) + EPS,
                                value=value,
                                name=name)
            else:
                return p.normal(loc=weights[:, :, 0:weights.shape[-1] // 2],
                                scale=softplus(weights[:, :, weights.shape[-1] // 2:]) + EPS,
                                value=value,
                                name=name)


        pos = p.normal(loc=ifcuda(torch.zeros((num_samples, batch_size, 2))),
                       scale=ifcuda(torch.ones((num_samples, batch_size, 2))),
                       value=q["pos"],
                       name="pos")

        pos_hidden = self.pos_hidden(pos)

        obs_targets = combined_normal('obs_targets', self.obs_targets_mean_std(
            pos_hidden))

        if not self.skip_obs_agents:
            obs_agents = combined_normal('obs_agents', self.obs_agents_mean_std(
                pos_hidden))
            z_mean_std = self.z_mean_std(torch.cat([pos_hidden,obs_agents,obs_targets] ,dim=-1))
        else:
            z_mean_std = self.z_mean_std(torch.cat([pos_hidden, obs_targets], dim=-1))

        if shuffledu is None:
            z = combined_normal('z',z_mean_std)
        else:
            z = combined_normal('z',z_mean_std,value=shuffledu)

        if not self.skip_obs_agents:
            message_mean = self.message_network(
                torch.cat([obs_agents, obs_targets, pos, z], dim=-1)
            )
        else:
            message_mean = self.message_network(
                torch.cat([obs_targets, pos, z], dim=-1)
            )

        if messages is not None:
            p.loss(binary_cross_entropy, message_mean, messages, name='messages')

        return p, torch.mean(message_mean,dim=0)


# loss function
def elbo(q, p,sample_dims=0):
    return disentanglement.libs.probtorch.probtorch.objectives.montecarlo.elbo(q, p, sample_dim=sample_dims, batch_dim=1,
                                                                             alpha=weights["A"],beta=weights["B"])

def compute_discrim_loss(Dz, Dz_perm, zeros, ones):
    return 0.5 * (cross_entropy(Dz, zeros) + cross_entropy(Dz_perm, ones))


def compute_discrim_loss_component(D):
    return weights["G"] * torch.mean(D[:, 0] - D[:, 1])
