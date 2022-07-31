import torch
from torch import relu
from torch.nn.functional import softplus

from disentanglement.libs.probtorch import probtorch
import torch.nn as nn
from disentanglement.libs.probtorch.probtorch.util import expand_inputs

from disentanglement.message_level.utils import expand_dicts
from disentanglement.timestep_level.timestep_model_parameters import *


def send2cuda(x):
    if CUDA:
        return x.cuda()


class Disentnanglement_Discriminator(nn.Module):
    def __init__(self, agent_behavior_size, num_agents, hidden_size=DISCRIMINATOR_SIZE):
        super(self.__class__, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(agent_behavior_size * num_agents, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, z):
        return self.network(z)


class Encoder(nn.Module):
    def __init__(self, num_agents, num_targets, obs_size, message_size, knowledge_size,
                 agent_behavior_size=AGENT_BEHAVIOR_SIZE, hidden_knowledge_size=ENCODER_HIDDEN_KNOWLEDGE_SIZE,
                 hidden_message_size=ENCODER_HIDDEN_MESSAGE_SIZE,
                 hidden1_size=ENCODER_HIDDEN1_SIZE, hidden2_size=ENCODER_HIDDEN2_SIZE,
                 hidden3_size=ENCODER_HIDDEN3_SIZE, hidden4_size=ENCODER_HIDDEN4_SIZE,
                 hidden5_size=ENCODER_HIDDEN5_SIZE):
        super(self.__class__, self).__init__()
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.obs_size = obs_size
        self.message_size = message_size
        self.agent_behavior_size = agent_behavior_size

        self.knowledge_hidden = nn.Sequential(
            nn.Linear(knowledge_size, hidden_knowledge_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_knowledge_size * 2, hidden_knowledge_size),
            nn.ReLU(),
            nn.Linear(hidden_knowledge_size, hidden_knowledge_size),
            nn.ReLU(),
        )

        self.positions_mean_std = nn.Sequential(
            nn.Linear(hidden_knowledge_size, hidden5_size * 2),
            nn.ReLU(),
            nn.Linear(hidden5_size * 2, hidden5_size),
            nn.ReLU(),
            nn.Linear(hidden5_size, 2 * 2 * num_agents),
        )

        self.message_mean_std = nn.Sequential(
            nn.Linear(hidden_knowledge_size + 2 * num_agents, hidden2_size * 2),
            nn.ReLU(),
            nn.Linear(hidden2_size * 2, hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size, 2 * message_size * num_agents)
        )
        self.messages_hidden = nn.Sequential(
            nn.Linear(message_size * num_agents, hidden_message_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_message_size * 2, hidden_message_size),
            nn.ReLU(),
            nn.Linear(hidden_message_size, hidden_message_size),
            nn.ReLU(),
        )

        self.obs_agents_mean_std = nn.Sequential(
            nn.Linear(hidden_message_size+message_size*num_agents, hidden4_size * 2),
            nn.ReLU(),
            nn.Linear(hidden4_size * 2, hidden4_size),
            nn.ReLU(),
            nn.Linear(hidden4_size, 2 * obs_size * num_agents),
        )

        self.obs_targets_mean_std = nn.Sequential(
            nn.Linear(hidden_message_size+message_size*num_agents, hidden4_size * 2),
            nn.ReLU(),
            nn.Linear(hidden4_size * 2, hidden4_size),
            nn.ReLU(),
            nn.Linear(hidden4_size, 2 * obs_size * num_agents),
        )

        self.targets_mean_std = nn.Sequential(
            nn.Linear(obs_size * num_agents + 2 * num_agents, hidden5_size * 2),
            nn.ReLU(),
            nn.Linear(hidden5_size * 2, hidden5_size),
            nn.ReLU(),
            nn.Linear(hidden5_size, 2 * 2 * num_targets),
        )
        self.agent_behavior_mean_std = nn.Sequential(
            nn.Linear(
                hidden_message_size + hidden_knowledge_size + 2 * num_agents  + 2 * obs_size * num_agents,
                hidden3_size * 2),
            nn.ReLU(),
            nn.Linear(hidden3_size * 2, hidden3_size),
            nn.ReLU(),
            nn.Linear(hidden3_size, hidden3_size),
            nn.ReLU(),
            nn.Linear(hidden3_size, 2 * agent_behavior_size * num_agents),
        )

    @expand_inputs
    @expand_dicts
    # We need num_samples for the parameter expansion to work correctly
    def forward(self, knowledge, batch_size, labels=None, num_samples=NUM_SAMPLES, extract_distributions=False):
        if labels is None:
            labels = {'positions': None, 'targets': None, 'obs_agents': None,
                      'obs_targets': None, 'messages': None, 'agent_behavior': None, 'our_position': None}

        q = probtorch.Trace()

        def combined_normal(name, weights):
            return q.normal(loc=torch.squeeze(torch.index_select(weights, -1, torch.tensor([0]).cuda()), dim=-1),
                            scale=softplus(torch.squeeze(torch.index_select(weights, -1, torch.tensor([1]).cuda()),
                                                         dim=-1)) + EPS,
                            value=labels[name],
                            name=name)

        knowledge_hidden = self.knowledge_hidden(knowledge)
        positions_mean_std = torch.reshape(self.positions_mean_std(knowledge_hidden),
                                           (num_samples, batch_size, self.num_agents, 2, 2))
        positions = combined_normal("positions", positions_mean_std)

        message_input = torch.cat(
            [knowledge_hidden, torch.flatten(positions, start_dim=2)], dim=-1)
        message_mean_std = torch.reshape(self.message_mean_std(message_input),
                                         (num_samples, batch_size, self.num_agents, self.message_size, 2))
        messages = combined_normal('messages', message_mean_std)
        messages_hidden = self.messages_hidden(torch.flatten(messages, start_dim=2))

        observation_inputs = torch.cat(
            [knowledge_hidden, torch.flatten(messages, start_dim=2)], dim=-1)

        obs_agents_mean_std = torch.reshape(self.obs_agents_mean_std(observation_inputs),
                                            (num_samples, batch_size, self.num_agents, self.obs_size, 2))
        obs_agents = combined_normal('obs_agents', obs_agents_mean_std)

        obs_targets_mean_std = torch.reshape(self.obs_targets_mean_std(observation_inputs),
                                             (num_samples, batch_size, self.num_agents, self.obs_size, 2))
        obs_targets = combined_normal('obs_targets', obs_targets_mean_std)

        targets_inputs = torch.cat(
            [torch.flatten(positions, start_dim=2), torch.flatten(obs_targets, start_dim=2)], dim=-1)
        targets_mean_std = torch.reshape(self.targets_mean_std(targets_inputs),
                                         (num_samples, batch_size, self.num_targets, 2, 2))
        targets = combined_normal("targets", targets_mean_std)

        agent_behavior_input = torch.cat(
            [knowledge_hidden, messages_hidden, torch.flatten(positions, start_dim=2),
             torch.flatten(obs_agents, start_dim=2),
             torch.flatten(obs_targets, start_dim=2)], dim=-1)
        agent_behavior_mean_std = torch.reshape(self.agent_behavior_mean_std(agent_behavior_input),
                                                (num_samples, batch_size, self.num_agents, self.agent_behavior_size, 2))
        agent_behavior = combined_normal('agent_behavior', agent_behavior_mean_std)

        if extract_distributions:
            dist_map = {
                "positions": positions_mean_std,
                "targets": targets_mean_std,
                "obs_targets": obs_targets_mean_std,
                "obs_agents": obs_agents_mean_std,
                "messages": message_mean_std,
                "agent_behavior": agent_behavior_mean_std,
            }
            for k in dist_map.keys():
                if len(dist_map[k].shape) == 4:
                    dist_map[k] = torch.stack([dist_map[k][:, :, :, 0],
                                              softplus(dist_map[k][:, :, :, 1]) + EPS],dim=-1)
                elif len(dist_map[k].shape) == 5:
                    dist_map[k] = torch.stack([dist_map[k][:, :, :,:, 0],
                                               softplus(dist_map[k][:, :, :, :,1]) + EPS], dim=-1)
                else:
                    print("WARNING, UNEXPECTED SHAPE IN DISTRIBUTION SIZE FOR KEY ", k)
            return q, dist_map

        return q


def binary_cross_entropy(x_mean, x, EPS=1e-9):
    return - (torch.log(x_mean + EPS) * x +
              torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1)


class Decoder(nn.Module):
    def __init__(self, num_agents, num_targets, obs_size, message_size, knowledge_size,
                 agent_behavior_size=AGENT_BEHAVIOR_SIZE, pos_hidden=DECODER_POSITION_HIDDEN_SIZE,
                 hidden1_size=DECODER_HIDDEN1_SIZE, hidden2_size=DECODER_HIDDEN2_SIZE,
                 hidden3_size=DECODER_HIDDEN3_SIZE):

        self.num_agents = num_agents
        self.num_targets = num_targets
        self.obs_size = obs_size
        self.message_size = message_size
        self.knowledge_size = knowledge_size
        self.agent_behavior_size = agent_behavior_size
        super(self.__class__, self).__init__()
        self.positions_hidden = nn.Sequential(
            nn.Linear(2 * num_agents, pos_hidden * 2),
            nn.ReLU(),
            nn.Linear(pos_hidden * 2, pos_hidden),
            nn.ReLU()
        )

        self.obs_agents_mean_std = nn.Sequential(
            nn.Linear(pos_hidden, hidden1_size * 2),
            nn.ReLU(),
            nn.Linear(hidden1_size * 2, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, 2 * obs_size * num_agents),
        )

        self.obs_targets_mean_std = nn.Sequential(
            nn.Linear(pos_hidden + num_targets * 2, hidden1_size * 2),
            nn.ReLU(),
            nn.Linear(hidden1_size * 2, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, 2 * obs_size * num_agents),
        )
        self.single_message_mean_std = nn.Sequential(
            nn.Linear(2 + obs_size + obs_size + agent_behavior_size, hidden2_size * 2),
            nn.ReLU(),
            nn.Linear(hidden2_size * 2, hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size, 2 * message_size),
        )

        self.agent_behavior_mean_std = nn.Sequential(
            nn.Linear(2*obs_size*num_agents+2*num_agents,hidden2_size*2),
            nn.ReLU(),
            nn.Linear(hidden2_size*2,hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size,agent_behavior_size*num_agents*2)
        )

        self.knowledge_network = nn.Sequential(
            nn.Linear(2*num_agents + message_size * num_agents, hidden3_size * 2),
            nn.ReLU(),
            nn.Linear(hidden3_size * 2, hidden3_size),
            nn.ReLU(),
            nn.Linear(hidden3_size, knowledge_size),
            nn.Sigmoid()
        )

    def forward(self, world_knowledge, batch_size, q=None, num_samples=NUM_SAMPLES):
        p = probtorch.Trace()

        if q is None:
            q = {'agent': None, 'positions': None, 'targets': None, 'obs_agents': None,
                 'obs_targets': None, 'messages': None, 'agent_behavior': None}

        def combined_normal(name, weights):
            return p.normal(loc=torch.squeeze(torch.index_select(weights, -1, torch.tensor([0]).cuda()), dim=-1),
                            scale=softplus(
                                torch.squeeze(torch.index_select(weights, -1, torch.tensor([1]).cuda()), dim=-1)) + EPS,
                            value=q[name],
                            name=name)

        positions = p.normal(loc=torch.zeros((num_samples, batch_size, self.num_agents, 2)).cuda(),
                             scale=1.5*torch.ones((num_samples, batch_size, self.num_agents, 2)).cuda(),
                             value=q["positions"],
                             name="positions")

        targets = p.normal(loc=torch.zeros((num_samples, batch_size, self.num_targets, 2)).cuda(),
                           scale=1.5*torch.ones((num_samples, batch_size, self.num_targets, 2)).cuda(),
                           value=q["targets"],
                           name="targets")



        positions_hidden = self.positions_hidden(
            torch.flatten(positions, start_dim=2))

        obs_agents_weights = torch.reshape(self.obs_agents_mean_std(positions_hidden),
                                           (num_samples, batch_size, self.num_agents, self.obs_size, 2))

        obs_targets_weights = torch.reshape(
            self.obs_targets_mean_std(torch.cat([positions_hidden, torch.flatten(targets, start_dim=2)], dim=-1)),
            (num_samples, batch_size, self.num_agents, self.obs_size, 2))

        obs_agents = combined_normal("obs_agents", obs_agents_weights)
        obs_targets = combined_normal("obs_targets", obs_targets_weights)


        agent_behavior_mean_std = torch.reshape(
        self.agent_behavior_mean_std(torch.cat([
            torch.flatten(positions,start_dim=2),
            torch.flatten(obs_agents,start_dim=2),
                                                torch.flatten(obs_targets,start_dim=2)],dim=-1)),
            (num_samples,batch_size,self.num_agents,self.agent_behavior_size,2))

        agent_behavior = combined_normal('agent_behavior', agent_behavior_mean_std)

        messages_mean_std = send2cuda(torch.zeros(num_samples, batch_size, self.num_agents, self.message_size, 2))

        for a in range(self.num_agents):
            this_obs_a = obs_agents[:, :, a, :]
            this_obs_t = obs_targets[:, :, a, :]
            this_behavior = agent_behavior[:, :, a, :]
            this_pos = positions[:, :, a, :]
            network_in = torch.flatten(torch.cat([this_pos, this_obs_a, this_obs_t, this_behavior], dim=-1),
                                       start_dim=2)
            network_out = self.single_message_mean_std(network_in)
            messages_mean_std[:, :, a, :, :] = torch.reshape(network_out,
                                                             (num_samples, batch_size, self.message_size, 2))

        messages = combined_normal("messages", messages_mean_std)


        mean_world_knowledge = self.knowledge_network(
            torch.cat([torch.flatten(positions,start_dim=2), torch.flatten(messages, start_dim=2)], dim=-1))

        if world_knowledge is not None:
            p.loss(binary_cross_entropy, world_knowledge, mean_world_knowledge, name='world_knowledge')

        return p, torch.mean(mean_world_knowledge, dim=0)


def compute_discrim_loss_component(D):
    return t_weights["G"] * torch.mean(D[:, 0] - D[:, 1])
