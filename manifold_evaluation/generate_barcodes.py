import json
import os
import pickle
import subprocess
import sys

import numpy
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import disentanglement.libs.disentanglement.gs as gs
import codecs


def get_model_pipe(name, num_obs, num_agents, agent, num_targets=None, message_size=None, allowerror=False):
    process = subprocess.Popen(('../encoder_io/get_model_pipe.sh', name, "agent" if agent else "timestep"),
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=None if allowerror else subprocess.PIPE)

    def terminate():
        process.stdin.close()
        process.terminate()
        process.wait(timeout=0.2)

    def read():
        while True:
            p = process.stdout.readline().decode("utf-8").strip()
            if p.startswith("return"):
                p = p.strip("return")
                decoded = codecs.decode(p[2:-3].replace("\\n", "").encode(), "base64")
                return pickle.loads(decoded)
            elif len(p) > 0:
                print(p)

    def write(x):
        pickled = codecs.encode(pickle.dumps(x), "base64")
        process.stdin.write(pickled + b'\n')
        process.stdin.flush()

    def callwithobj(obj):
        write(obj)
        return read()

    def encoder(message):
        return callwithobj((True, message, None))

    def decoder(compressed_qs, batch_size=None):
        if agent:
            qs = {'pos': compressed_qs[:, 0:2],
                  'n_covered_targets': None,
                  'obs_agents': compressed_qs[:, 2:2 + num_obs],
                  'obs_targets': compressed_qs[:, 2 + num_obs:2 + 2 * num_obs],
                  'z': compressed_qs[:, 2 + 2 * num_obs:]}
        else:
            qs = {'positions': compressed_qs[:, :num_agents * 2].reshape((num_agents, 2)),
                  'targets': compressed_qs[:,
                             num_agents * 2:
                             num_agents * 2 + num_targets * 2].reshape((num_targets, 2)),
                  'obs_agents': compressed_qs[:,
                                num_agents * 2 + num_targets * 2:
                                num_agents * 2 + num_targets * 2 + num_agents * num_obs].reshape((num_agents, num_obs)),
                  'obs_targets': compressed_qs[:,
                                 num_agents * 2 + num_targets * 2 + num_agents * num_obs:
                                 num_agents * 2 + num_targets * 2 + num_agents * num_obs * 2].reshape(
                      (num_agents, num_obs)),
                  'messages': compressed_qs[:,
                              num_agents * 2 + num_targets * 2 + num_agents * num_obs * 2:
                              num_agents * 2 + num_targets * 2 + num_agents * num_obs * 2 + num_agents * message_size].reshape(
                      (num_agents, message_size)),
                  'agent_behavior': compressed_qs[:,
                                    num_agents * 2 + num_targets * 2 + num_agents * num_obs * 2 + num_agents * message_size:
                                    -num_agents].reshape(
                      (num_agents, -1)),
                  'agent': compressed_qs[:, -num_agents:]
                  }

        return callwithobj((False, qs, batch_size))

    return encoder, decoder, terminate


def compare_embedding_spaces_fake(params, z_size, L_0, gamma, name, agent, plot=False):
    num_samples = 256 * 8
    batch_size = 256
    values_per_dim = 10
    num_batches = num_samples // batch_size
    obs_size = params["obs_size"]
    num_agents = params["num_agents"]
    results_dir = params["results_dir"]
    message_size = params["message_size"]
    num_targets = params["num_targets"]

    encoder, decoder, terminator = get_model_pipe(name, obs_size, num_agents, agent, num_targets, message_size,
                                                  allowerror=True)

    if agent:
        factor_keys = ['pos',
                       'obs_agents',
                       'obs_targets',
                       'z']
        nz = 2 * obs_size + 2 + z_size + num_agents
    else:
        factor_keys = ['positions',
                       'targets',
                       'obs_agents',
                       'obs_targets',
                       'messages',
                       'agent_behavior',
                       'agent']
        nz = num_targets * 2 + num_agents * (1 + 2 + obs_size * 2 + message_size + z_size)

    samples = np.random.randn(num_samples, nz)
    agent_samples = np.eye(num_agents)[np.random.choice(num_agents, size=num_samples)]

    samples[:, -num_agents:] = agent_samples
    samples = torch.Tensor(samples)

    assert num_samples / num_batches == num_samples // num_batches, f'num samples needs to be divisible by num batches'
    results_dict = dict([(i, {}) for i in range(nz - num_agents + 1)])

    if not agent:
        cur_factor = nz - num_agents
        # Agent
        print("Computing agent factors")
        for cur_value in range(num_agents):
            embeds = []
            for b in tqdm(range(num_batches)):
                sample = samples[batch_size * b:batch_size * (b + 1)]
                zz = sample.clone()
                for i in range(num_agents):
                    if i == cur_value:
                        zz[:, -num_agents + i] = 1
                    else:
                        zz[:, -num_agents + i] = 0
                obs = decoder(zz.view(batch_size, -1), batch_size=batch_size)

                embed = []
                embedding_dict = encoder(obs)
                for k in factor_keys:
                    embed.append(embedding_dict[k].numpy())

                embeds.append(numpy.concatenate(embed, axis=1))

            embeds = np.concatenate(embeds, axis=0)
            rlts = gs.rlts(embeds, L_0=L_0, gamma=gamma, n=100)
            mrlt = np.mean(rlts, axis=0)
            if plot:
                gs.fancy_plot(mrlt, label=f'MRLT of {cur_factor}_{cur_value}')
                # plt.xlim([0, 30])
                plt.legend()
                plt.savefig(f"{results_dir}/plots/embedding_space_fake_{name}_{cur_factor}_{cur_value}.png")
                plt.close()
            results_dict[cur_factor][cur_value] = rlts.tolist()

    for cur_factor in range(nz - (0 if agent else num_agents)):
        print("Computing factor", cur_factor)
        for _ in range(values_per_dim):
            cur_value = np.random.randn(1).item()
            embeds = []
            for b in tqdm(range(num_batches)):
                sample = samples[batch_size * b:batch_size * (b + 1)]
                zz = sample.clone()
                zz[:, cur_factor] = cur_value
                obs = decoder(zz.view(batch_size, -1), batch_size=batch_size)

                embed = []
                embedding_dict = encoder(obs)
                for k in factor_keys:
                    embed.append(embedding_dict[k].numpy())

                embeds.append(numpy.concatenate(embed, axis=1))

            embeds = np.concatenate(embeds, axis=0)
            rlts = gs.rlts(embeds, L_0=L_0, gamma=gamma, n=100)
            mrlt = np.mean(rlts, axis=0)
            if plot:
                gs.fancy_plot(mrlt, label=f'MRLT of {cur_factor}_{cur_value}')
                # plt.xlim([0, 30])
                plt.legend()
                plt.savefig(f"{results_dir}/plots/embedding_space_fake_{name}_{cur_factor}_{cur_value}.png")
                plt.close()
            results_dict[cur_factor][cur_value] = rlts.tolist()

    # Write to file
    with open(params["results_file"], "w") as f:
        json.dump(results_dict, f)
    print(f'Done')
    terminator()


if __name__ == '__main__':
    name = "medium"
    params = {
        "obs_size": 16,
        "num_agents": 5,
        "results_dir": "./results",
        "results_file": "./results/barcode/" + name + "_results.json",
        "num_targets": 5,
        "message_size": 10
    }
    agent=True
    z_size = 6
    compare_embedding_spaces_fake(params, z_size, 100, 1 / 128, name,agent, plot=True)
