import gc
import os
import pickle
import random
import time

import numpy as np

from disentanglement.message_level import train as message_level_train
from disentanglement.message_level import models as message_level_models

from disentanglement.message_level import load_data
from disentanglement.message_level.model_parameters import MESSAGE_SIZE
from disentanglement.timestep_level.timestep_model_parameters import *
import disentanglement.message_level.utils as u
from disentanglement.timestep_level import timestep_models as models
from disentanglement.timestep_level.timestep_models import compute_discrim_loss_component


def train(data, enc, dec, discriminator, optimizer, discrim_optimizer, params,
          label_mask=None, label_fraction=LABEL_FRACTION, sample_size=NUM_SAMPLES):
    # torch.autograd.set_detect_anomaly(True)
    epoch_elbo = 0.0
    enc.train()
    dec.train()
    discriminator.train()
    N = 0
    elbo_log = []
    reconstruction_log = []
    discrim_log = []
    start_time = time.time()

    for b, (knowledge, labels) in enumerate(data):
        if b > TRAIN_CUTOFF:
            break
        batch_size = knowledge.size(0)
        N += batch_size
        u.alltocuda(labels)
        knowledge = u.ifcuda(knowledge)

        zeros = u.ifcuda(torch.zeros(batch_size, dtype=torch.long))
        ones = u.ifcuda(torch.ones(batch_size, dtype=torch.long))

        labels['agent_behavior'] = None

        q = enc(knowledge, batch_size, labels, num_samples=sample_size)
        agent_behavior = torch.mean(q["agent_behavior"].value, dim=0)

        p, reconstructed_knowledge = dec(knowledge, batch_size, q=q, num_samples=sample_size)
        D = discriminator(torch.flatten(agent_behavior, start_dim=1))
        discrim_loss_component = compute_discrim_loss_component(D)
        loss = -message_level_models.elbo(q, p) + discrim_loss_component
        reconstruction_distance = torch.mean(torch.linalg.norm(
            knowledge - reconstructed_knowledge, dim=-1))

        if loss == torch.inf:
            print("Skipping batch ", b, "due to infinite loss")
            continue

        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        reconstruction_log.append(reconstruction_distance.item())
        elbo_log.append(loss.item() - discrim_loss_component.item())
        epoch_elbo -= loss.item()

        # Train discriminator
        D_z_perm = discriminator(u.permute_dims(torch.flatten(agent_behavior, start_dim=1).detach()))
        discrim_loss = message_level_models.compute_discrim_loss(D, D_z_perm, zeros, ones)
        discrim_optimizer.zero_grad()
        discrim_loss.backward()
        discrim_log.append(discrim_loss.item())

        if b % 100 == 0:
            print("batch", b, "of", len(data), "loss", loss.item() - discrim_loss_component.item(), "discrim",
                  discrim_loss_component.item(), "discrim_loss", discrim_loss.item(),
                  "time per batch:", (time.time() - start_time) / (b + 1))

        optimizer.step()
        discrim_optimizer.step()

        u.alltocpu(labels)

    return epoch_elbo / N, label_mask, elbo_log, reconstruction_log, discrim_log


def test(data, enc, dec,computecat=False):
    enc.eval()
    dec.eval()
    if CUDA:
        enc.cuda()
        dec.cuda()

    elbo = []
    errors = {
        'obs_agents': [],
        'obs_targets': [],
        'knowledge': [],
        'messages': [],
        'positions': [],
        'targets': []

    }
    labelcatlog = {'obs_targets':[]}
    predcatlog  = {'obs_targets':[]}

    labelcatlog['obs_agents'] = []
    predcatlog['obs_agents'] = []

    N = 0
    for b, (knowledge, labels) in enumerate(data):

        if b > TRAIN_CUTOFF:
            break

        u.alltocuda(labels)

        labelcatlog['obs_agents'].append(torch.flatten(labels['obs_agents']>-0.1).cpu().detach())
        labelcatlog['obs_targets'].append(torch.flatten(labels['obs_targets']>-0.1).cpu().detach())

        knowledge = u.ifcuda(knowledge)
        batch_size = knowledge.size()[0]
        N += batch_size
        q, distributions = enc(knowledge, batch_size, num_samples=1, extract_distributions=True)
        p, reconstructed_knowledge = dec(knowledge, batch_size, q, num_samples=1)

        elbo.append(-message_level_models.elbo(q, p).item())

        for k in distributions.keys():
            if len(distributions[k].shape) > 4:
                distributions[k] = torch.mean(distributions[k][:, :, :, :, 0], dim=0)
            else:
                distributions[k] = distributions[k][:, :, :, 0]

        errors['obs_agents'].append(
            torch.linalg.norm(distributions['obs_agents'] - labels['obs_agents'], dim=-1).cpu().detach().numpy())

        errors['obs_targets'].append(
            torch.linalg.norm(distributions['obs_targets'] - labels['obs_targets'], dim=-1).cpu().detach().numpy())

        predcatlog['obs_agents'].append(torch.flatten(distributions['obs_agents']>-0.1).cpu().detach())

        predcatlog['obs_targets'].append(torch.flatten(distributions['obs_targets']>-0.1).cpu().detach())


        errors['messages'].append(
            torch.linalg.norm(distributions['messages'] - labels['messages'], dim=-1).cpu().detach().numpy())

        errors['positions'].append(
            torch.linalg.norm(distributions['positions'] - labels['positions'], dim=-1).cpu().detach().numpy())
        errors['targets'].append(
            torch.linalg.norm(distributions['targets'] - labels['targets'], dim=-1).cpu().detach().numpy())

        errors['knowledge'].append(
            torch.linalg.norm(reconstructed_knowledge.squeeze() - knowledge, dim=-1).cpu().detach().numpy())
        u.alltocpu(labels)

    for k in errors.keys():
        errors[k] = np.concatenate(errors[k], axis=0).flatten()
    for k in labelcatlog.keys():
        labelcatlog[k] = torch.concat(labelcatlog[k], dim=0).int()
    for k in predcatlog.keys():
        predcatlog[k] = torch.concat(predcatlog[k], dim=0).int()
    if computecat:
        return elbo, errors, labelcatlog, predcatlog

    return elbo, errors


def train_model(epochs, checkpoint_path=None, sample_size=NUM_SAMPLES, continue_data=None,dotest=False):
    if not os.path.isdir(DATA_PATH):
        os.makedirs(DATA_PATH)

    train_data, test_data, params = load_data.get_data(BATCH_SIZE, NUM_WORKERS, agent=False, save_name=SAVE_NAME)

    print("Parameters:", params)

    if continue_data is not None:
        enc, dec, discriminator, elbo_log, reconstruction_log, discrim_log, start_from = continue_data
        start_from = start_from + 1

    else:
        enc = models.Encoder(params['num_agents'], params['num_targets'], params['obs_size'], MESSAGE_SIZE,
                             KNOWLEDGE_SIZE,
                             agent_behavior_size=AGENT_BEHAVIOR_SIZE,
                             hidden_knowledge_size=ENCODER_HIDDEN_KNOWLEDGE_SIZE,
                             hidden_message_size=ENCODER_HIDDEN_MESSAGE_SIZE,
                             hidden1_size=ENCODER_HIDDEN1_SIZE, hidden2_size=ENCODER_HIDDEN2_SIZE,
                             hidden3_size=ENCODER_HIDDEN3_SIZE, hidden4_size=ENCODER_HIDDEN4_SIZE,
                             hidden5_size=ENCODER_HIDDEN5_SIZE)
        dec = models.Decoder(params['num_agents'], params['num_targets'], params['obs_size'], MESSAGE_SIZE,
                             KNOWLEDGE_SIZE,
                             agent_behavior_size=AGENT_BEHAVIOR_SIZE, pos_hidden=DECODER_POSITION_HIDDEN_SIZE,
                             hidden1_size=DECODER_HIDDEN1_SIZE, hidden2_size=DECODER_HIDDEN2_SIZE,
                             hidden3_size=DECODER_HIDDEN3_SIZE)

        discriminator = models.Disentnanglement_Discriminator(AGENT_BEHAVIOR_SIZE, params['num_agents'],
                                                              hidden_size=DISCRIMINATOR_SIZE)

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

    mask = {}
    print("Beginning training...")

    print(sum([sum(p.numel() for p in model.parameters() if p.requires_grad) for model in [enc, dec, discriminator]]),
          "parameters")

    total_start = time.time()
    for e in range(start_from, epochs):
        train_start = time.time()
        train_elbo, mask, elbo_log_2, reconstruction_log_2, discrim_log_2 = train(train_data, enc, dec, discriminator,
                                                                                  optimizer, discrim_optimizer, params,
                                                                                  mask, LABEL_FRACTION,
                                                                                  sample_size=sample_size)
        elbo_log = elbo_log + elbo_log_2
        reconstruction_log = reconstruction_log + reconstruction_log_2
        discrim_log = discrim_log + discrim_log_2
        if checkpoint_path is not None:
            message_level_train.save_all(checkpoint_path,
                                         "checkpoint_" + str(e) + "_model.pkl", "checkpoint" + str(e) + "_logs.pkl",
                                         enc, dec, discriminator, elbo_log, reconstruction_log, discrim_log)

        train_end = time.time()
        if dotest:
            test_start = time.time()
            test_elbo, test_accuracy = test(test_data, enc, dec)
            test_end = time.time()

            print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e (%ds)' % (
                e, train_elbo, train_end - train_start,
                np.mean(test_elbo), test_end - test_start))
        else:
            print('[Epoch %d] Train: ELBO %.4e (%ds)' % (
                e, train_elbo, train_end - train_start))

    print("Trained in %ds" % (time.time() - total_start))
    return enc, dec, discriminator, elbo_log, reconstruction_log, discrim_log


def get_models(savedir, modelname, training_data_name, load=True, epochs=None, checkpoint_path=None, load_from=None):
    if checkpoint_path is not None:
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

            with open(os.path.join(checkpoint_path, "checkpoint" + str(load_from) + "_logs.pkl"), "rb") as f:
                elbo_log, reconstruction_log, discrim_log = pickle.load(f)

            continue_data = (enc, dec, discrim, elbo_log, reconstruction_log, discrim_log, load_from)
        else:
            continue_data = None
        enc, dec, discrim, elbo_log, reconstruction_log, discrim_log = train_model(epochs, checkpoint_path,
                                                                                   sample_size=NUM_SAMPLES,
                                                                                   continue_data=continue_data)

        message_level_train.save_all(savedir, modelname, "training_data_" + SAVE_NAME + ".pkl", enc, dec, discrim,
                                     elbo_log, reconstruction_log, discrim_log)

    return enc, dec, discrim, elbo_log, reconstruction_log, discrim_log


if __name__ == '__main__':
    savedir = "saved_models"
    modelname = "pickled_model_" + SAVE_NAME + "_timestep.pkl"
    checkpoint_path = os.path.join("saved_models", "checkpoints")
    training_data_name = "training_data_" + SAVE_NAME + "_timestep.pkl"
    load = False
    epochs = 20
    load_from = None

    get_models(savedir, modelname, training_data_name, load, epochs, checkpoint_path, load_from)
