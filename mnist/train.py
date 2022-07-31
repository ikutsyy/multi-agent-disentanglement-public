import random

from model_parameters import *
import models
import torch

from torchvision import datasets, transforms
import os

import time


def train(data, enc, dec, optimizer, label_fraction=LABEL_FRACTION):
    epoch_elbo = 0.0
    enc.train()
    dec.train()
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size(0) == NUM_BATCH:
            N += NUM_BATCH
            images = images.view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(NUM_BATCH, NUM_DIGITS)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
            if CUDA:
                images = images.cuda()
                labels_onehot = labels_onehot.cuda()
            optimizer.zero_grad()
            q = enc(images, labels_onehot, num_samples=NUM_SAMPLES)
            p = dec(images, q, num_samples=NUM_SAMPLES, batch_size=NUM_BATCH)
            loss = -models.elbo(q, p, bias=BIAS_TRAIN)
            loss.backward()
            optimizer.step()
            if CUDA:
                loss = loss.cpu()
            epoch_elbo -= loss.item()
    return epoch_elbo / N


def test(data, enc, dec, infer=True):
    enc.eval()
    dec.eval()
    epoch_elbo = 0.0
    epoch_correct = 0
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == NUM_BATCH:
            N += NUM_BATCH
            images = images.view(-1, NUM_PIXELS)
            if CUDA:
                images = images.cuda()
            q = enc(images, num_samples=NUM_SAMPLES)
            p = dec(images, q, num_samples=NUM_SAMPLES, batch_size=NUM_BATCH)
            batch_elbo = models.elbo(q, p, bias=BIAS_TEST)
            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.detach().numpy()

            log_p = p.log_joint(0, 1)
            log_q = q.log_joint(0, 1)
            log_w = log_p - log_q
            w = torch.nn.functional.softmax(log_w, 0)
            y_samples = q['y'].value
            y_expect = (w.unsqueeze(-1) * y_samples).sum(0)
            _, y_pred = y_expect.detach().max(-1)
            if CUDA:
                y_pred = y_pred.cpu()
            epoch_correct += (labels == y_pred).float().sum()
    return epoch_elbo / N, epoch_correct / N


def get_data(batch_size):
    train_data = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_data = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    return train_data, test_data


def train_model(epochs):
    if not os.path.isdir(DATA_PATH):
        os.makedirs(DATA_PATH)

    train_data, test_data = get_data(NUM_BATCH)

    enc = models.Encoder()
    dec = models.Decoder()
    if CUDA:
        enc.cuda()
        dec.cuda()
    optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                                 lr=LEARNING_RATE)

    for e in range(epochs):
        train_start = time.time()
        train_elbo = train(train_data, enc, dec,
                                 optimizer, LABEL_FRACTION)
        train_end = time.time()
        test_start = time.time()
        test_elbo, test_accuracy = test(test_data, enc, dec)
        test_end = time.time()

        print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, Accuracy %0.3f (%ds)' % (
            e, train_elbo, train_end - train_start,
            test_elbo, test_accuracy, test_end - test_start))

    return enc, dec
