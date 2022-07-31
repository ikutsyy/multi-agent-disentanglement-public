import gc
import json
import os
import pickle
import random
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from disentanglement.message_level.utils import ifprint, np_sigmoid, display_top
from disentanglement.message_level.model_parameters import *


# def call_init(x):
#     info = torch.utils.data.get_worker_info()
#     print(info)
#     info.dataset.dataset.set_order()

class AgentDataset(Dataset):
    def __init__(self, data_dir, params, shuffle):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.params = params
        self.n_agents = params["num_agents"]
        self.chunk_size = params["chunk_size"] * self.n_agents
        self.num_chunks = len([1 for filename in os.listdir(data_dir) if "agent.npz" in filename])

        self.order = None
        self.id = None
        self.num_workers = None
        self.chunk_order = None
        self.chunk = None
        self.chunk_index = None
        self.index = None

    def do_setup(self):
        info = torch.utils.data.get_worker_info()
        seed = hash(info.seed) % (2 ** 32 - 1)
        np.random.seed(seed)
        if self.shuffle:
            self.order = np.random.permutation(self.chunk_size)
        else:
            if info.num_workers > 1:
                print("WARNING: unshuffled only supports one worker")
            self.order = np.arange(0, stop=self.chunk_size)
        self.id = info.id
        self.chunk_index = 0
        self.index = 0
        self.num_workers = info.num_workers
        self.chunk_order = np.arange(0, self.num_chunks)[self.id::self.num_workers]
        if self.shuffle:
            np.random.shuffle(self.chunk_order)
        if self.num_chunks % self.num_workers != 0:
            print("WARNING: Correct epoch requires the number of chunks to be divisble by workers, have",
                  self.num_workers, "workers and ", self.num_chunks, "chunks")
        loaded = np.load(os.path.join(self.data_dir, str(self.chunk_order[self.chunk_index]) + "agent.npz"))
        self.chunk = {}
        for k in loaded.keys():
            self.chunk[k] = torch.from_numpy(loaded[k])
        self.chunk["message"] = torch.sigmoid(self.chunk["message"])
        if PREPROCESS_OBS:
            self.chunk["obs_agents"][self.chunk["obs_agents"] == 0.0] = -1
            self.chunk["obs_targets"][self.chunk["obs_targets"] == 0.0] = -1

    def __getitem__(self, index):
        if self.chunk is None:
            self.do_setup()
        if self.index == self.chunk_size - 1:
            # print("Moving onto next chunk...")
            self.index = 0
            if self.shuffle:
                np.random.shuffle(self.order)
            self.chunk_index = 0
            self.chunk_index += 1
            if self.chunk_index == len(self.chunk_order):
                if self.shuffle:
                    np.random.shuffle(self.chunk_order)
                self.chunk_index = 0
            loaded = np.load(os.path.join(self.data_dir, str(self.chunk_order[self.chunk_index]) + "agent.npz"))
            self.chunk = {}
            for k in loaded.keys():
                self.chunk[k] = torch.from_numpy(loaded[k])
            self.chunk["message"] = torch.sigmoid(self.chunk["message"])
            if PREPROCESS_OBS:
                self.chunk["obs_agents"][self.chunk["obs_agents"] == 0.0] = -1
                self.chunk["obs_targets"][self.chunk["obs_targets"] == 0.0] = -1

        i = self.order[self.index]
        agent = i % self.n_agents


        x = torch.concat([self.chunk["message"][i], torch.sigmoid(self.chunk["pos"][i])], dim=-1)
        y = dict(zip(["agent", "pos", "vel", "obs_agents", "obs_targets","chunk"],
                     [agent,
                      self.chunk["pos"][i],
                      self.chunk["vel"][i],
                      self.chunk["obs_agents"][i],
                      self.chunk["obs_targets"][i],
                      self.chunk_index]))
        self.index += 1

        return x, y

    def __len__(self):
        return self.num_chunks * self.chunk_size


class TimestepDataset(Dataset):
    def __init__(self, data_dir, params, shuffle):
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.params = params
        self.n_agents = params["num_agents"]
        self.chunk_size = params["chunk_size"] * self.n_agents
        self.num_chunks = len([1 for filename in os.listdir(data_dir) if "timestep.npz" in filename])
        self.order = None
        self.id = None
        self.num_workers = None
        self.chunk_order = None
        self.chunk = None
        self.chunk_index = None
        self.index = None

    def do_setup(self):
        info = torch.utils.data.get_worker_info()
        seed = hash(info.seed) % (2 ** 32 - 1)
        np.random.seed(seed)
        if self.shuffle:
            self.order = np.random.permutation(self.chunk_size)
        else:
            if info.num_workers > 1:
                print("WARNING: unshuffled only supports one worker")
            self.order = np.arange(0, stop=self.chunk_size)
        self.id = info.id
        self.chunk_index = 0
        self.index = 0
        self.num_workers = info.num_workers
        self.chunk_order = np.arange(0, self.num_chunks)[self.id::self.num_workers]
        if self.shuffle:
            np.random.shuffle(self.chunk_order)
        loaded = np.load(os.path.join(self.data_dir, str(self.chunk_order[self.chunk_index]) + "timestep.npz"))
        if self.num_chunks % self.num_workers != 0:
            print("WARNING: Correct epoch requires the number of chunks to be divisble by workers, have",
                  self.num_workers, "workers and ", self.num_chunks, "chunks")
        self.chunk = {}
        for k in loaded.keys():
            self.chunk[k] = torch.from_numpy(loaded[k])
        self.chunk["world_knowledge"] = torch.sigmoid(self.chunk["world_knowledge"])
        self.chunk["targets"] = torch.squeeze(self.chunk["targets"], dim=1)
        if PREPROCESS_OBS:
            self.chunk["obs_agents"][self.chunk["obs_agents"] == 0.0] = -1
            self.chunk["obs_targets"][self.chunk["obs_targets"] == 0.0] = -1

    def __getitem__(self, index):
        if self.chunk is None:
            self.do_setup()
        if self.index == self.chunk_size - 1:
            print("Moving onto next chunk...")
            self.index = 0
            if self.shuffle:
                np.random.shuffle(self.order)
            self.chunk_index = 0
            self.chunk_index += 1
            if self.chunk_index == len(self.chunk_order):
                if self.shuffle:
                    np.random.shuffle(self.chunk_order)
                self.chunk_index = 0
            loaded = np.load(os.path.join(self.data_dir, str(self.chunk_order[self.chunk_index]) + "timestep.npz"))
            self.chunk = {}
            for k in loaded.keys():
                self.chunk[k] = torch.from_numpy(loaded[k])
            self.chunk["world_knowledge"] = torch.sigmoid(self.chunk["world_knowledge"])
            self.chunk["targets"] = torch.squeeze(self.chunk["targets"], dim=1)
            if PREPROCESS_OBS:
                self.chunk["obs_agents"][self.chunk["obs_agents"] == 0.0] = -1
                self.chunk["obs_targets"][self.chunk["obs_targets"] == 0.0] = -1

        index = self.order[self.index]
        i = index // self.n_agents
        self.index += 1
        agent = index % self.n_agents

        x = self.chunk["world_knowledge"][i][agent]

        # Order agents by distance since IDs are unknown to the knowledge tensor
        _,order = torch.sort(torch.linalg.norm(self.chunk["positions"][i]-self.chunk["positions"][i][agent],dim=-1),descending=False)
        _,target_order = torch.sort(torch.linalg.norm(self.chunk["targets"][i]-self.chunk["positions"][i][agent],dim=-1),descending=False)

        y = dict(zip(["agent", "targets", "positions", "velocities", "obs_agents", "obs_targets", "messages"],
                     [agent,
                      self.chunk["targets"][i][target_order],
                      self.chunk["positions"][i][order],
                      self.chunk["velocities"][i][order],
                      self.chunk["obs_agents"][i][order],
                      self.chunk["obs_targets"][i][order],
                      torch.concat([self.chunk["messages"][i], self.chunk["positions"][i]], dim=-1)[order]
                      ]))

        return x, y

    def __len__(self):
        return self.num_chunks * self.chunk_size


def preprocess_episode(df, num_agents):
    firstrow = df.iloc[0]
    num_steps = (len(df) // num_agents) - 1
    agent_data = {
        "message": np.ones((num_agents * num_steps, len(firstrow["gnn_in_features"]))),
        "agent": np.ones(num_agents * num_steps),
        "pos": np.ones((num_agents * num_steps, 2)),
        "vel": np.ones((num_agents * num_steps, 2)),
        "obs_agents": np.ones((num_agents * num_steps, len(firstrow["obs_agents"]))),
        "obs_targets": np.ones((num_agents * num_steps, len(firstrow["obs_targets"]))),
    }

    timestep_data = {
        "targets": np.ones((num_steps, *np.asarray(firstrow["target_positions"]).shape)),
        "messages": np.ones((num_steps, num_agents, len(firstrow["gnn_in_features"]))),
        "positions": np.ones((num_steps, num_agents, 2)),
        "velocities": np.ones((num_steps, num_agents, 2)),
        "obs_agents": np.ones((num_steps, num_agents, len(firstrow["obs_agents"]))),
        "obs_targets": np.ones((num_steps, num_agents, len(firstrow["obs_targets"]))),
        "world_knowledge": np.ones((num_steps, num_agents, len(firstrow["total_local_knowledge"]))),
    }

    for sim_step in range(0, (len(df) // num_agents) - 1):
        i = sim_step
        for agent in range(num_agents):
            ai = i * num_agents + agent
            index = sim_step * num_agents + agent
            row = df.iloc[index]
            agent_data["message"][ai] = np.asarray(df.iloc[index + num_agents].gnn_in_features)
            timestep_data["messages"][i][agent] = agent_data["message"][ai]

            if agent == 0:
                timestep_data["targets"][i] = np.asarray(row.target_positions)

            agent_data["agent"][ai] = row.agent
            agent_data["pos"][ai] = np.asarray([row.px, row.py])
            agent_data["vel"][ai] = np.asarray([row.vx, row.vy])

            timestep_data["positions"][i][agent] = agent_data["pos"][ai]
            timestep_data["velocities"][i][agent] = agent_data["vel"][ai]

            agent_data["obs_agents"][ai] = np.asarray(row.obs_agents)
            agent_data["obs_targets"][ai] = np.asarray(row.obs_targets)

            timestep_data["obs_agents"][i][agent] = agent_data["obs_agents"][ai]
            timestep_data["obs_targets"][i][agent] = agent_data["obs_targets"][ai]
            timestep_data["world_knowledge"][i][agent] = np.asarray(df.iloc[index + num_agents].total_local_knowledge)

    return timestep_data, agent_data


def preprocess_file(data_path, params_path, save_name, multi_file, i):
    print("Processing dataset", i)

    with open(data_path, "rb") as f, open(params_path, "r") as p:
        df = pickle.load(f)
        params = json.load(p)
        num_agents = params['env_config']['n_agents']

        timesteps_by_ep = []
        agents_by_ep = []

        episode_nums = set(df.episode.values)

        if 'total_local_knowledge' not in df.columns:
            print("Could not find total local knowledge in", data_path)
            print("Columns are", df.columns)
            return

        for ep in episode_nums:
            subset = df[df.episode == ep]
            if ep % 10 == 0:
                print("Processing episode", ep, "of " + str(len(episode_nums)) + " with",
                      len(subset) // num_agents, "steps")
            b, c = preprocess_episode(subset, num_agents)
            timesteps_by_ep.append(b)
            agents_by_ep.append(c)
        del df

        agents = {}
        timesteps = {}

        for k in agents_by_ep[0].keys():
            agents[k] = np.concatenate([x[k] for x in agents_by_ep], dtype=np.float32)

        del agents_by_ep

        for k in timesteps_by_ep[0].keys():
            timesteps[k] = np.concatenate([x[k] for x in timesteps_by_ep], dtype=np.float32)

        del timesteps_by_ep

        prefix = str(i) if multi_file else ""

        print("Saving...")
        np.savez(os.path.join(DATA_PATH, save_name, prefix + "agent"), **agents)
        np.savez(os.path.join(DATA_PATH, save_name, prefix + "timestep"), **timesteps)


def preprocess_data(save_name, multi_file_directory, multi_file=True):
    params_path = PARAMS_FILE

    if multi_file:
        data_paths = [os.path.join(multi_file_directory, name) for name in os.listdir(multi_file_directory)]
    else:
        data_paths = [os.path.join(DATA_PATH, DATASET_NAME)]

    if not os.path.exists(os.path.join(DATA_PATH, save_name)):
        os.mkdir(os.path.join(DATA_PATH, save_name))

    args = [(data_path, params_path, save_name, multi_file, i) for i, data_path in enumerate(data_paths)]
    pool = multiprocessing.Pool(processes=PREPROCESS_CORES)
    pool.starmap(preprocess_file, args)


def get_data(batch_size, num_workers=NUM_WORKERS, agent=True, save_name=SAVE_NAME, print_info=True, shuffle=True):
    params_path = PARAMS_FILE
    chunk_size = CHUNK_SIZE_MAP[save_name]
    if not os.path.exists(os.path.join(DATA_PATH, save_name, "agents.pkl")) \
            and not os.path.exists(os.path.join(DATA_PATH, save_name, "0agent.npz")) \
            and not os.path.exists(os.path.join(DATA_PATH, save_name, "train", "0agent.npz")):
        print("Could not find",os.path.join(DATA_PATH, save_name))
        ifprint(print_info, "Preprocessing dataset for runtime efficiency -- only happens on first run...")
        preprocess_data(save_name, multi_file_directory=MULTI_FILE_DIRECTORY, multi_file=True)
        ifprint(print_info, "Processed")

    with open(params_path, "r") as p:
        ifprint(print_info, "Loading Dataset '" + save_name + "'")
        start_time = time.time()

        params = json.load(p)
        num_agents = params['env_config']['n_agents']
        num_targets = params['env_config']['n_targets']
        obs_size = params['env_config']['obs_vector_size']

        our_params = dict(zip(["num_agents", "num_targets", "obs_size", "message_size", "batch_size",
                               "chunk_size"],
                              [num_agents, num_targets, obs_size, MESSAGE_SIZE, batch_size, chunk_size]))

        if agent:
            data_dir = os.path.join(DATA_PATH, save_name)
            trainset = AgentDataset(data_dir=os.path.join(data_dir, "train"), params=our_params, shuffle=shuffle)
            testset = AgentDataset(data_dir=os.path.join(data_dir, "test"), params=our_params, shuffle=shuffle)

            ifprint(print_info, "Loaded in {0} seconds".format((time.time() - start_time)))

            ifprint(print_info, "Using dataset of size", len(trainset))

            return DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH, drop_last=True,
                              persistent_workers=PERSISTENT_WORKERS), \
                   DataLoader(testset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH, drop_last=True,
                              persistent_workers=PERSISTENT_WORKERS), our_params

        else:
            data_dir = os.path.join(DATA_PATH, save_name)
            trainset = TimestepDataset(data_dir=os.path.join(data_dir, "train"), params=our_params, shuffle=shuffle)
            testset = TimestepDataset(data_dir=os.path.join(data_dir, "test"), params=our_params, shuffle=shuffle)

            our_params["knowledge_size"] = KNOWLEDGE_SIZE
            ifprint(print_info, "Loaded in {0} seconds".format((time.time() - start_time)))

            ifprint(print_info, "Using dataset of size", len(trainset))

            return DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH, drop_last=True,
                              persistent_workers=PERSISTENT_WORKERS), \
                   DataLoader(testset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH, drop_last=True,
                              persistent_workers=PERSISTENT_WORKERS), our_params


if __name__ == '__main__':
    preprocess_data("val", multi_file=True, multi_file_directory="../../data/val_multi")
