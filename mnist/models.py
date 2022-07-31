import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
from torch.nn import Parameter
import sys
from disentanglement.libs.probtorch import probtorch
from disentanglement.libs.probtorch.probtorch.util import expand_inputs


from model_parameters import *


class Encoder(nn.Module):

    def __init__(self, num_pixels=NUM_PIXELS,
                 num_hidden1=NUM_HIDDEN1,
                 num_hidden2=NUM_HIDDEN2,
                 num_style=NUM_STYLE,
                 num_digits=NUM_DIGITS):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(
            nn.Linear(num_pixels, num_hidden1),
            nn.ReLU())
        self.digit_log_weights = nn.Linear(num_hidden1, num_digits)
        self.digit_temp = 0.66
        self.style_mean = nn.Sequential(
            nn.Linear(num_hidden1 + num_digits, num_hidden2),
            nn.ReLU(),
            nn.Linear(num_hidden2, num_style))
        self.style_log_std = nn.Sequential(
            nn.Linear(num_hidden1 + num_digits, num_hidden2),
            nn.ReLU(),
            nn.Linear(num_hidden2, num_style))

    @expand_inputs
    def forward(self, images, labels=None, num_samples=NUM_SAMPLES):
        q = probtorch.Trace()
        hidden = self.enc_hidden(images)
        digits = q.concrete(logits=self.digit_log_weights(hidden),
                            temperature=self.digit_temp,
                            value=labels,
                            name='y')

        hidden2 = torch.cat([digits, hidden], -1)
        styles_mean = self.style_mean(hidden2)

        styles_std = self.style_log_std(hidden2).exp()
        q.normal(loc=styles_mean,
                 scale=styles_std,
                 name='z')
        return q


def binary_cross_entropy(x_mean, x, EPS=1e-9):
    return - (torch.log(x_mean + EPS) * x +
              torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1)


class Decoder(nn.Module):
    def __init__(self, num_pixels=NUM_PIXELS,
                 num_hidden1=NUM_HIDDEN1,
                 num_hidden2=NUM_HIDDEN2,
                 num_style=NUM_STYLE,
                 num_digits=NUM_DIGITS):
        super(self.__class__, self).__init__()
        self.dec_hidden = nn.Sequential(
            nn.Linear(num_style + num_digits, num_hidden2),
            nn.ReLU(),
            nn.Linear(num_hidden2, num_hidden1),
            nn.ReLU())
        self.num_style = num_style
        self.num_digits = num_digits
        self.digit_temp = 0.66
        self.dec_images = nn.Sequential(
            nn.Linear(num_hidden1, num_pixels),
            nn.Sigmoid())

    def forward(self, images, q=None, num_samples=NUM_SAMPLES, batch_size=NUM_BATCH):
        p = probtorch.Trace()
        digit_log_weights = torch.zeros(num_samples, batch_size, self.num_digits)
        style_mean = torch.zeros(num_samples, batch_size, self.num_style)
        style_std = torch.ones(num_samples, batch_size, self.num_style)

        if CUDA:
            digit_log_weights = digit_log_weights.cuda()
            style_mean = style_mean.cuda()
            style_std = style_std.cuda()


        digits = p.concrete(logits=digit_log_weights,
                                     temperature=self.digit_temp,
                                     value=q['y'],
                                     name='y')

        styles = p.normal(loc=style_mean,
                          scale=style_std,
                          value=q['z'],
                          name='z')


        hiddens = self.dec_hidden(torch.cat([digits, styles], -1))
        images_mean = self.dec_images(hiddens)
        p.loss(binary_cross_entropy, images_mean, images, name='images')
        return p

# loss function
def elbo(q, p, alpha=weights["A"], beta=weights["B"], bias=1.0):
    return probtorch.objectives.montecarlo.elbo(q, p,alpha=1,beta=1)