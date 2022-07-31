import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.nn.functional import softplus

from disentanglement.message_level import utils, models, load_data
from disentanglement.message_level.load_data import get_data
from disentanglement.message_level.utils import alltocuda, ifcuda, alltocpu, display_top
from disentanglement.message_level.model_parameters import *


def save_all(path, name, training_data_name, enc, dec, discriminator, elbo_log, reconstruction_log, discrim_log):
    with open(os.path.join(path, name), "wb") as f:
        pickle.dump((enc, dec, discriminator), f)

    with open(os.path.join(path, training_data_name), "wb") as f:
        pickle.dump((elbo_log, reconstruction_log, discrim_log), f)


def train(data, enc, dec, discriminator, optimizer, discrim_optimizer, params,
          allowed_chunks=None, sample_size=NUM_SAMPLES, checkpoint_path=None):
    if DEBUG:
        torch.autograd.set_detect_anomaly(True)

    print("alpha", weights["A"], "beta", weights["B"], "gamma", weights["G"], "batch size", BATCH_SIZE, "num_samples",
          sample_size,
          "chunk mask length", ("None" if allowed_chunks is None else len(allowed_chunks)), "training cutoff",
          TRAIN_CUTOFF)
    epoch_elbo = 0.0
    enc.train()
    dec.train()
    discriminator.train()
    N = 0
    elbo_log = []
    reconstruction_log = []
    discrim_log = []

    start_time = time.time()

    for b, (message, labels) in enumerate(data):
        if b >= TRAIN_CUTOFF:
            break
        if b % TRAIN_SAVE_BATCHES == TRAIN_SAVE_BATCHES - 1 and checkpoint_path is not None:
            save_all(checkpoint_path, "saving_in_batch_" + str(b) + "_model.pkl",
                     "saving_in_batch_" + str(b) + "_logs.pkl",
                     enc, dec, discriminator, elbo_log, reconstruction_log, discrim_log)

        batch_size = message.shape[0]
        N += batch_size
        alltocuda(labels)
        message = ifcuda(message)

        zeros = ifcuda(torch.zeros(batch_size, dtype=torch.long))
        ones = ifcuda(torch.ones(batch_size, dtype=torch.long))

        labels['z'] = None

        if allowed_chunks is not None and labels['chunk'] not in allowed_chunks:
            q = enc(message)
            this_z = q["z"].value

        else:
            q = enc(message, labels, num_samples=sample_size)
            this_z = torch.mean(q["z"].value, dim=0)

        p, reconstructed_message = dec(message, q, batch_size=batch_size, num_samples=sample_size)
        D = discriminator(this_z)
        discrim_loss_component = models.compute_discrim_loss_component(D)
        loss = -models.elbo(q, p) + discrim_loss_component
        reconstruction_distance = torch.mean(torch.linalg.norm(
            torch.logit(message) - torch.logit(reconstructed_message), dim=-1))

        if loss == torch.inf:
            print("Skipping batch ", b, "due to infinite loss")
            continue

        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        reconstruction_log.append(reconstruction_distance.item())
        elbo_log.append(loss.item() - discrim_loss_component.item())
        epoch_elbo -= loss.item()

        # Train discriminator
        D_z_perm = discriminator(utils.permute_dims(this_z.detach()))
        discrim_loss = models.compute_discrim_loss(D, D_z_perm, zeros, ones)
        discrim_optimizer.zero_grad()
        discrim_loss.backward()
        discrim_log.append(discrim_loss.item())

        if b % 100 == 0:
            print("batch", b, "of", len(data), "loss", loss.item() - discrim_loss_component.item(), "discrim",
                  discrim_loss_component.item(), "discrim_loss", discrim_loss.item(),
                  "time per batch:", (time.time() - start_time) / (b + 1))

        optimizer.step()
        discrim_optimizer.step()

        alltocpu(labels)

    return epoch_elbo / N, allowed_chunks, elbo_log, reconstruction_log, discrim_log


def test(data, enc, dec, skip_obs_agents=False,computecat=False,shuffleu=False,ratedist=False):

    enc.eval()
    dec.eval()
    if CUDA:
        enc.cuda()
        dec.cuda()

    print(sum(p.numel() for p in dec.parameters() if p.requires_grad),
          "parameters")

    elbo = []
    errors = {
        'pos': [],
        'obs_targets': [],
        'message': [],
    }
    if not skip_obs_agents:
        errors['obs_agents'] = []

    labelcatlog = {'obs_targets':[]}
    predcatlog  = {'obs_targets':[]}

    if not skip_obs_agents:
        labelcatlog['obs_agents'] = []
        predcatlog['obs_agents'] = []
    N = 0
    R = 0
    D = 0
    start_time = time.time()
    for b, (message, labels) in enumerate(data):
        if b >= TRAIN_CUTOFF:
            break
        alltocuda(labels)
        if not skip_obs_agents:
           labelcatlog['obs_agents'].append(torch.flatten(labels['obs_agents']>-0.1).cpu().detach())
        labelcatlog['obs_targets'].append(torch.flatten(labels['obs_targets']>-0.1).cpu().detach())

        message = ifcuda(message)
        batch_size = message.size()[0]
        N += batch_size
        q, distributions = enc(message, num_samples=1, extract_distributions=True)
        if shuffleu:
            shuffledu = q["z"].value[:,torch.randperm(q["z"].value.size()[1])]
        else:
            shuffledu = None
        p, reconstructed_messages = dec(message, q, batch_size=batch_size, num_samples=1,shuffledu=shuffledu)

        for k in distributions.keys():
            if len(distributions[k].shape) > 3:
                distributions[k] = torch.mean(distributions[k][:, :, :, 0], dim=0)
            else:
                distributions[k] = distributions[k][:, :, 0]

        elbo.append(-models.elbo(q, p).item())
        errors['pos'].append(torch.linalg.norm(distributions['pos'] - labels['pos'], dim=-1).cpu().detach().numpy())

        if not skip_obs_agents:
            errors['obs_agents'].append(torch.linalg.norm(distributions['obs_agents'] - labels['obs_agents'],
                                                          dim=-1).cpu().detach().numpy())
            predcatlog['obs_agents'].append(torch.flatten(distributions['obs_agents']>-0.1).cpu().detach())

        predcatlog['obs_targets'].append(torch.flatten(distributions['obs_targets']>-0.1).cpu().detach())
        errors['obs_targets'].append(
            torch.linalg.norm(distributions['obs_targets'] - labels['obs_targets'], dim=-1).cpu().detach().numpy())

        errors['message'].append(
            torch.linalg.norm(reconstructed_messages - message, dim=-1).cpu().detach().numpy())

        if ratedist:
            r,d = utils.get_rate_distortion(p,q)
            R += r
            D += d

        if b % 100 == 0:
            print("batch", b, "of", len(data),
                  "time per batch:", (time.time() - start_time) / (b + 1))


        alltocpu(labels)
    for k in errors.keys():
        errors[k] = np.concatenate(errors[k], axis=0)
    for k in labelcatlog.keys():
        labelcatlog[k] = torch.concat(labelcatlog[k], dim=0).int()
    for k in predcatlog.keys():
        predcatlog[k] = torch.concat(predcatlog[k], dim=0).int()


    if computecat:
        return elbo,errors,labelcatlog,predcatlog
    if ratedist:
        return elbo,errors,R/len(data),D/len(data)
    return elbo, errors


def train_model(epochs, checkpoint_path=None, sample_size=NUM_SAMPLES, skip_obs_agents=False, do_test=True,
                allowed_chunks=None, continue_data=None):
    if not os.path.isdir(DATA_PATH):
        os.makedirs(DATA_PATH)

    train_data, test_data, params = load_data.get_data(BATCH_SIZE, NUM_WORKERS)
    print("Parameters:", params)

    if continue_data is not None:
        enc, dec, discriminator, elbo_log, reconstruction_log, discrim_log, start_from = continue_data
        start_from = start_from + 1

    else:
        enc = models.Encoder(params['num_agents'], params['message_size'], params['obs_size'], z_size=Z_SIZE,
                             hidden_message_size=ENCODER_HIDDEN1_SIZE,
                             hidden2_size=ENCODER_HIDDEN2_SIZE, hidden3_size=ENCODER_HIDDEN3_SIZE,
                             hidden4_size=ENCODER_HIDDEN4_SIZE, skip_obs_agents=skip_obs_agents)
        dec = models.Decoder(params['num_agents'], params['message_size'], params['obs_size'], z_size=Z_SIZE,
                             pos_hidden_size=DECODER_HIDDEN1_SIZE,
                             z_hidden_size=DECODER_HIDDEN2_SIZE, obs_hidden_size=DECODER_HIDDEN3_SIZE,
                             message_hidden_size=DECODER_HIDDEN4_SIZE, skip_obs_agents=skip_obs_agents)

        discriminator = models.Disentnanglement_Discriminator(z_size=Z_SIZE, hidden_size=DISCRIMINATOR_SIZE)

        elbo_log = []
        reconstruction_log = []
        discrim_log = []

        start_from = 0

    if CUDA:
        enc.cuda()
        dec.cuda()
        discriminator.cuda()

    optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                                 lr=LEARNING_RATE)

    discrim_optimizer = torch.optim.Adam(list(discriminator.parameters()),
                                         lr=LEARNING_RATE)

    print(sum([sum(p.numel() for p in model.parameters() if p.requires_grad) for model in [enc, dec, discriminator]]),
          "parameters")
    print(sum(p.numel() for p in enc.parameters() if p.requires_grad),
          "encoder parameters")
    print("Beginning training...")
    total_start = time.time()
    for e in range(start_from, epochs):
        train_start = time.time()
        train_elbo, _, elbo_log_2, reconstruction_log_2, discrim_log_2 = train(train_data, enc, dec, discriminator,
                                                                               optimizer, discrim_optimizer, params,
                                                                               allowed_chunks=allowed_chunks,
                                                                               sample_size=sample_size,
                                                                               checkpoint_path=checkpoint_path)
        elbo_log = elbo_log + elbo_log_2
        reconstruction_log = reconstruction_log + reconstruction_log_2
        discrim_log = discrim_log + discrim_log_2
        if checkpoint_path is not None:
            save_all(checkpoint_path, "checkpoint_" + str(e) + "_model.pkl", "checkpoint_" + str(e) + "_logs.pkl", enc,
                     dec, discriminator, elbo_log, reconstruction_log, discrim_log)

        train_end = time.time()
        if do_test:
            test_start = time.time()
            test_elbo, test_accuracy = test(test_data, enc, dec, skip_obs_agents=skip_obs_agents)
            test_end = time.time()

            print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e (%ds)' % (
                e, train_elbo, train_end - train_start,
                np.mean(test_elbo), test_end - test_start))
        else:
            print('[Epoch %d] Train: ELBO %.4e (%ds) ' % (
                e, train_elbo, train_end - train_start))

    print("Trained in %ds" % (time.time() - total_start))
    return enc, dec, discriminator, elbo_log, reconstruction_log, discrim_log


def get_models(savedir, modelname, training_data_name, load=True, epochs=None, checkpoint_path=None,
               skip_obs_agents=False, allowed_chunks=None, load_from=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(savedir, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    model_path = os.path.join(savedir, modelname)
    train_data_path = os.path.join(savedir, training_data_name)
    if load:
        with open(model_path, "rb") as f:
            enc, dec, discrim = pickle.load(f)

        with open(train_data_path, "rb") as f:
            elbo_log, reconstruction_log, discrim_log = pickle.load(f)
    else:
        if load_from is not None:
            print("Loading", os.path.join(checkpoint_path, "checkpoint_" + str(load_from) + "_model.pkl"))
            with open(os.path.join(checkpoint_path, "checkpoint_" + str(load_from) + "_model.pkl"), "rb") as f:
                enc, dec, discrim = pickle.load(f)
            with open(os.path.join(checkpoint_path, "checkpoint_" + str(load_from) + "_logs.pkl"), "rb") as f:
                elbo_log, reconstruction_log, discrim_log = pickle.load(f)
            continue_data = (enc, dec, discrim, elbo_log, reconstruction_log, discrim_log, load_from)
        else:
            continue_data = None
        enc, dec, discrim, elbo_log, reconstruction_log, discrim_log = train_model(epochs, checkpoint_path,
                                                                                   sample_size=NUM_SAMPLES,
                                                                                   skip_obs_agents=skip_obs_agents,
                                                                                   allowed_chunks=allowed_chunks,
                                                                                   continue_data=continue_data)
        save_all(savedir, modelname, training_data_name, enc, dec, discrim, elbo_log, reconstruction_log, discrim_log)

    return enc, dec, discrim, elbo_log, reconstruction_log, discrim_log


# if __name__ == '__main__':
#     savedir = "saved_models"
#     modelname = "aaa"+SAVE_NAME+"_.pkl"
#     training_data_name = "aaa"+SAVE_NAME+"_.pkl"
#     epochs=2
#     get_models(savedir,modelname,training_data_name,load=False,epochs=1,skip_obs_agents=True)
