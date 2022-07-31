import os
import pickle
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from disentanglement.direct_model_flat import direct_model
from disentanglement.message_level import load_data
from disentanglement.message_level.model_parameters import DATA_PATH
from disentanglement.message_level.utils import alltocuda, ifcuda, alltocpu
from disentanglement.direct_model_flat.direct_model_parameters import *

def save(path,name,training_data_name,model,log):
    with open(os.path.join(path, name), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(path,training_data_name ), "wb") as f:
        pickle.dump(log, f)

def train(data, model, optimizer, params):


    model.train()
    if CUDA:
        model.cuda()
    N = 0
    loss_log = []

    start_time = time.time()

    for b, (message, labels) in enumerate(data):

        batch_size = message.shape[0]
        N += batch_size
        alltocuda(labels)
        message = ifcuda(message)

        predicted = model(message)
        loss = model.compute_loss(predicted,labels)

        if loss == torch.inf:
            print("Skipping batch ",b,"due to infinite loss")
            continue

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_log.append(loss.item())

        if b % 100 == 0:
            print("batch", b, "of", len(data), "loss", loss.item(), "time per batch:", (time.time()-start_time)/(b+1))

        alltocpu(labels)

    return loss_log


def test(data, model,computecat=False):
    model.eval()
    if CUDA:
        model.cuda()

    loss = []
    errors = {
        'pos': [],
        'obs_targets': [],
        'obs_agents': [],
    }

    labelcatlog = {'obs_targets': []}
    predcatlog = {'obs_targets': []}

    labelcatlog['obs_agents'] = []
    predcatlog['obs_agents'] = []
    N = 0

    for b, (message, labels) in enumerate(data):
        alltocuda(labels)
        labelcatlog['obs_agents'].append(torch.flatten(labels['obs_agents'] > -0.1).cpu().detach())
        labelcatlog['obs_targets'].append(torch.flatten(labels['obs_targets'] > -0.1).cpu().detach())

        message = ifcuda(message)
        batch_size = message.size()[0]
        N += batch_size
        q = model(message)

        loss.append(model.compute_loss(q,labels).item())
        errors['pos'].append(torch.linalg.norm(q['pos'] - labels['pos'], dim=-1).cpu().detach().numpy())

        errors['obs_agents'].append(torch.linalg.norm(q['obs_agents'] - labels['obs_agents'],
                                                      dim=-1).cpu().detach().numpy())
        predcatlog['obs_agents'].append(torch.flatten(q['obs_agents'] > -0.1).cpu().detach())

        predcatlog['obs_targets'].append(torch.flatten(q['obs_targets'] > -0.1).cpu().detach())
        errors['obs_targets'].append(
            torch.linalg.norm(q['obs_targets'] - labels['obs_targets'], dim=-1).cpu().detach().numpy())
        alltocpu(labels)
    for k in errors.keys():
        errors[k] = np.concatenate(errors[k], axis=0)
    for k in labelcatlog.keys():
        labelcatlog[k] = torch.concat(labelcatlog[k], dim=0).int()
    for k in predcatlog.keys():
        predcatlog[k] = torch.concat(predcatlog[k], dim=0).int()

    if computecat:
        return loss, errors, labelcatlog, predcatlog
    return loss, errors


def train_model(epochs, data_name, checkpoint_path=None,continue_data=None):
    if not os.path.isdir(DATA_PATH):
        os.makedirs(DATA_PATH)

    train_data, test_data, params = load_data.get_data(BATCH_SIZE, num_workers=NUM_WORKERS, agent=True, save_name=data_name)
    print("Parameters:", params)

    if continue_data is not None:
        model, loss_log, start_from = continue_data
        start_from = start_from+1
    else:
        model = direct_model.DirectPredictor(params['num_agents'], params['message_size'], params['obs_size'])
        loss_log = []
        start_from=0

    if CUDA:
        model.cuda()

    optimizer = torch.optim.Adam(list(model.parameters()), lr=LEARNING_RATE)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")

    print("Beginning training...")
    total_start = time.time()
    for e in range(start_from,epochs):
        train_start = time.time()
        loss_log += train(train_data,model,optimizer,params)

        if checkpoint_path is not None:
            save(checkpoint_path,"checkpoint_"+str(e)+"_model.pkl","checkpoint"+str(e)+ "_logs.pkl",model,loss_log)

        train_end = time.time()

        print('[Epoch %d] Train: Loss %.4e (%ds)' % (
            e, sum(loss_log[-len(train_data)//4:])/(len(train_data)//4), train_end - train_start))

    print("Trained in %ds" % (time.time() - total_start))
    return model,  loss_log


def get_models(savedir, modelname,training_data_name,data_name, load=True, epochs=None,checkpoint_path = None,load_from=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(savedir,"checkpoints")
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
    model_path = os.path.join(savedir, modelname)
    train_data_path = os.path.join(savedir, training_data_name)
    if load:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(train_data_path, "rb") as f:
            loss_log = pickle.load(f)
    else:
        print("Loading", os.path.join(checkpoint_path, "checkpoint_" + str(load_from) + "_model.pkl"))
        if load_from is not None:
            with open(os.path.join(checkpoint_path, "checkpoint_" + str(load_from) + "_model.pkl"), "rb") as f:
                model = pickle.load(f)
            with open(os.path.join(checkpoint_path, "checkpoint" + str(load_from) + "_logs.pkl"), "rb") as f:
                loss_log = pickle.load(f)
            continue_data = (model,loss_log, load_from)
        else:
            continue_data = None
        model,loss_log = train_model(epochs,data_name,checkpoint_path,continue_data=continue_data)
        save(savedir, modelname,training_data_name,model,loss_log)

    return model,loss_log

if __name__ == '__main__':
    savedir = "saved_models"
    modelname = "partial_pickled_model_" + 'large' + "_direct.pkl"
    checkpoint_path = os.path.join("saved_models","partial_checkpoints")
    data_name = "partial"
    training_data_name = "partial_training_data_" + 'large' + "_direct.pkl"
    load = False
    epochs = 20
    model,loss = get_models(savedir, modelname,training_data_name,data_name, load, epochs,checkpoint_path)