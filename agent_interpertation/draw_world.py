import json
import math
import sys

import matplotlib
import pandas
import seaborn
from torch.multiprocessing import Pool, Process, set_start_method

from disentanglement.agent_interpertation.timestep_correlation import possible_knowledge_indexes
from disentanglement.direct_timestep_model import train_direct as timestep_direct_train, error_prediction_train as timestep_error_train

from disentanglement.direct_model import direct_model, train_direct, error_prediction_train
from disentanglement.timestep_level import timestep_train
from disentanglement.agent_interpertation.render import render, render_on_perception, WHITE, PURPLE, BLUE, dist_from, \
    ORANGE, RED, render_just_targets, BROWN

try:
    set_start_method('spawn')
except RuntimeError:
    pass
import os

import matplotlib.pyplot as plt
import numpy
import torch

import pygame
import numpy as np
import cv2

from disentanglement.message_level import model_parameters
from disentanglement.message_level.load_data import get_data
from disentanglement.message_level import utils as u
from disentanglement.message_level import train as message_train

X = 1
Y = 0


def draw_episode_with_predictions(model, data_dir, chunk, e, save_file, agent, cutoff, all_agents=False,
                                  draw_messages=False, num_samples=32, skipped_obs_agents=False,separatemodels=False):
    loaded = np.load(os.path.join(data_dir, str(chunk) + "timestep.npz"))
    chunk = {}
    if cutoff is None:
        cutoff = 499
    else:
        cutoff = min(499, cutoff)
    for k in loaded.keys():
        chunk[k] = torch.from_numpy(loaded[k])
    chunk["targets"] = torch.squeeze(chunk["targets"])
    chunk["messages"] = torch.sigmoid(chunk["messages"])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    display = None

    def getr(message):
        if separatemodels:
            r = {}
            predicted = model[0](message)
            errors = model[1](message,dosqrt=True)
            for k in predicted.keys():
                r[k] = torch.stack([predicted[k],torch.nan_to_num(errors[k],0.01)],dim=-1)
        else:
            _, r = model(message, num_samples=num_samples, extract_distributions=True)
        return r

    j = e * 500

    if all_agents:
        reconstructed = []
        for agent in range(chunk["positions"].shape[1]):
            message = torch.concat(
                [chunk["messages"][j:j + 499, agent], torch.sigmoid(chunk["positions"][j:j + 499, agent])],
                dim=-1).cuda()
            r = getr(message)
            u.alltocpu(r)
            u.alldetatch(r)

            if len(r["pos"].shape > 3):
                for k in r:
                    r[k] = torch.mean(r[k], dim=0)
            if skipped_obs_agents:
                r["obs_agents"] = None
            reconstructed.append(r)

    else:
        message = torch.concat(
            [chunk["messages"][j:j + 499, agent], torch.sigmoid(chunk["positions"][j:j + 499, agent])], dim=-1).cuda()
        reconstructed = getr(message)
        if len(reconstructed["pos"].shape) > 3:
            for k in reconstructed:
                reconstructed[k] = reconstructed[k][0]
        u.alltocpu(reconstructed)
        u.alldetatch(reconstructed)

    for i in range(cutoff):
        robots = chunk["positions"][j + i]
        targets = chunk["targets"][j + i]
        if all_agents:
            display, cfg, dim = render(robots, targets, mode="build", display=display)
        else:
            if draw_messages:
                display, cfg, dim = render(robots, targets, mode="build", display=display,
                                           message=chunk["messages"][j + i][agent])
            else:
                display, cfg, dim = render(robots, targets, mode="build", display=display)

        if all_agents:
            for agent in range(chunk["positions"].shape[1]):
                display = render_on_perception(display, reconstructed[agent]["pos"][i],
                                               reconstructed[agent]["obs_agents"][i],
                                               reconstructed[agent]["obs_targets"][i], cfg, dim)
        else:
            display = render_on_perception(display, reconstructed["pos"][i],
                                           None if skipped_obs_agents else reconstructed["obs_agents"][i],
                                           reconstructed["obs_targets"][i], cfg, dim)

        raw = pygame.surfarray.array3d(display)

        print(i)
        if out is None:
            out = cv2.VideoWriter(save_file, fourcc, 30, raw.shape[1::-1])
        out.write(np.flip(np.flip(raw, axis=1), axis=2))

    print("finished")


def draw_world_predicted_episode(model, data_dir, chunk, e, save_file, agent, cutoff=None,
                                 num_samples=8, draw_zs=False,separatemodels=False):
    if cutoff is None:
        cutoff = 499
    else:
        cutoff = min(499, cutoff)
    loaded = np.load(os.path.join(data_dir, str(chunk) + "timestep.npz"))
    chunk = {}
    for k in loaded.keys():
        chunk[k] = torch.from_numpy(loaded[k])
    chunk["world_knowledge"] = torch.sigmoid(chunk["world_knowledge"])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    display = None

    def getr(knowledge):
        if separatemodels:
            r = {}
            predicted = model[0](knowledge)
            errors = model[1](knowledge,dosqrt=True)
            for k in predicted.keys():
                r[k] = torch.stack([predicted[k],torch.nan_to_num(errors[k],0.01)],dim=-1)
        else:
            _, r = model(knowledge, batch_size=499, num_samples=num_samples, extract_distributions=True)
        return r

    j = e * 500

    knowledge = chunk["world_knowledge"][j:j + 499, agent].cuda()
    reconstructed = getr(knowledge)
    if len(reconstructed["positions"].shape) > 4:
        for k in reconstructed:
            reconstructed[k] = torch.mean(reconstructed[k], dim=0)
    x = reconstructed['obs_agents'][:, :, 0, 0].detach().cpu().numpy()

    u.alltocpu(reconstructed)
    u.alldetatch(reconstructed)

    for i in range(cutoff):
        if display is not None:
            display.fill(WHITE)
        # Only draw agents we have any info about

        _,order = torch.sort(torch.linalg.norm(chunk["positions"][j + i]-chunk["positions"][j + i][agent],dim=-1),descending=False)
        _,target_order = torch.sort(torch.linalg.norm(chunk["targets"][j + i].squeeze()-chunk["positions"][j + i][agent],dim=-1),descending=False)

        robots = chunk["positions"][j + i]
        targets = chunk["targets"][j + i].squeeze()

        possagents,posstargets = possible_knowledge_indexes(torch.unsqueeze(robots,0),torch.unsqueeze(targets,0))
        possagents = possagents[0]
        posstargets = posstargets[0]

        display, cfg, dim = render(robots, targets, mode="build", display=display)


        for a in possagents:
            if a == agent:
                agent_color = PURPLE
                target_predictions = reconstructed["targets"][i]
                display = render_just_targets(target_predictions,ORANGE,BROWN,255,display,posstargets)
            else:
                agent_color = RED
                target_predictions = None
            pos = reconstructed["positions"][i][a]
            display, cfg, dim = render_on_perception(display, pos,
                                                     reconstructed["obs_agents"][i][a],
                                                     reconstructed["obs_targets"][i][a], cfg, dim,target_predictions=target_predictions,
                                                     return_multiple=True, agent_color=agent_color,target_color=ORANGE,lowconfhue=255)

        raw = pygame.surfarray.array3d(display)

        print(i)
        if out is None:
            out = cv2.VideoWriter(save_file, fourcc, 30, raw.shape[1::-1])
        out.write(np.flip(np.flip(raw, axis=1), axis=2))

    print("finished")

    pygame.quit()
    out.release()


def run_test():
    _, test_data, _ = get_data(batch_size=1, num_workers=1, agent=False, save_name="medium", shuffle=False)
    savefile = os.path.join("render_dir", "test.jpg")
    videodir = os.path.join("render_dir", "video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    display = None

    for b, (_, rest) in enumerate(test_data):
        robots = rest["positions"][0]
        targets = rest["targets"][0]
        raw, display = render(robots, targets, savefile=savefile, mode="rgb_array", display=display)
        print(b)
        if out is None:
            out = cv2.VideoWriter(videodir, fourcc, 30, raw.shape[1::-1])
        out.write(np.flip(np.flip(raw, axis=1), axis=2))
        if b > 30:
            break
    pygame.quit()
    out.release()


def do_animation(savedir, prefix, modelname, training_data_name, data_dir, chunk, episode, agent, cutoff,
                 all_agents=False, messages=False, skipped_obs_agents=False, adversary=False,savedir_error=None):

    print(savedir_error)
    if savedir_error is None:
        model, _, _, _, _, _ = message_train.get_models(savedir, modelname, training_data_name, load=True)
    else:
        predictor,_ = train_direct.get_models(savedir,"pickled_model_large_direct.pkl","training_data_large_direct.pkl",
                                            None,load=True)
        error_predictor,_ = error_prediction_train.get_models(savedir,None,"error_pickled_model_large_direct.pkl",
                                                            "error_training_data_large_direct.pkl",None,load=True)
        model = (predictor,error_predictor)

    draw_episode_with_predictions(model, data_dir, chunk, episode,
                                  "render_dir/" + prefix + (
                                      "adversary_" if adversary else "cooperative_") + "prediction_video_" + str(
                                      chunk) + "_" + str(episode) + "_" + str(
                                      agent) + ".avi", agent, cutoff, all_agents=all_agents, draw_messages=messages,
                                  skipped_obs_agents=skipped_obs_agents,separatemodels=(savedir_error is not None))



def do_world_animation(savedir, prefix, modelname, training_data_name, data_dir, chunk, episode, agent, cutoff,
                       adversary=False,savedir_error=None):
    if savedir_error is None:
        model, _, _, _, _, _ = timestep_train.get_models(savedir, modelname, training_data_name, load=True)
    else:
        predictor,_ = timestep_direct_train.get_models(savedir, "pickled_model_large_direct.pkl", "training_data_large_direct.pkl",
                                            None, load=True)
        error_predictor,_ = timestep_error_train.get_models(savedir, None, "error_pickled_model_large_direct.pkl",
                                                            "error_training_data_large_direct.pkl", None, load=True)
        model = (predictor, error_predictor)
    draw_world_predicted_episode(model, data_dir, chunk, episode,
                                 "render_dir/" + prefix + (
                                     "adversary_" if adversary else "cooperative_") + "world_prediction_video_" + str(
                                     chunk) + "_" + str(episode) + "_" + str(
                                     agent) + ".avi", agent, cutoff, draw_zs=False,separatemodels=(savedir_error is not None))


if __name__ == '__main__':
    # savedir = "../message_level/saved_models/checkpoints"
    # modelname = "checkpoint_3_model.pkl"
    # training_data_name = "checkpoint_3_logs.pkl"

    adversary = True
    if adversary:
        data_dir = "../../data/adversarial/test"
    else:
        data_dir = "../../data/val/test"

    skipped_obs_agents = False
    chunk = int(sys.argv[1])
    start_episode = 0
    agent = 0
    cutoff = 100000000000
    num_episodes = 5
    world = ("w" in sys.argv[2])
    if adversary and world:
        agent = 1

    use_mlp = True

    prefix = "skipped_obs" if skipped_obs_agents else ""
    savedir2 = None if not use_mlp else True
    if world:
        if use_mlp:
            savedir = "../direct_timestep_model/saved_models/"
            modelname = ""
            training_data_name = ""
        else:
            savedir = "../timestep_level/saved_models/checkpoints/"
            modelname = "checkpoint_5_model.pkl"
            training_data_name = "checkpoint5_logs.pkl"
    else:
        if use_mlp:
            savedir = "../direct_model/saved_models/"
            modelname = ""
            training_data_name = ""
        elif skipped_obs_agents:
            savedir = "../message_level/saved_models"
            modelname = "noobs_pickled_model_large.pkl"
            training_data_name = "noobs_training_data_large.pkl"

        else:
            savedir = "../message_level/saved_models"
            modelname = "pickled_model_large.pkl"
            training_data_name = "training_data_large.pkl"



    if world:
        for i in range(num_episodes):
            do_world_animation(savedir, prefix, modelname, training_data_name, data_dir, chunk, start_episode + i,
                               agent, cutoff, adversary=adversary,savedir_error=savedir2)
    else:
        for i in range(num_episodes):
            do_animation(savedir, prefix, modelname, training_data_name, data_dir, chunk, start_episode + i, agent,
                         cutoff,
                         all_agents=False, messages=False, skipped_obs_agents=skipped_obs_agents, adversary=adversary,savedir_error=savedir2)
