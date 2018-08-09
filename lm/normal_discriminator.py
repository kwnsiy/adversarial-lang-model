# -*- coding: utf-8 -*-

import os
import random

import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        # if self.training:
        return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        # return din

class Discriminator(nn.Module):
    def __init__(self, num_classes, const_classes, vocab_size, emb_dim, filter_sizes, num_filters, dropout, padding_idx):
        super(Discriminator, self).__init__()
        self.emb_src = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.convs_src = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(sum(num_filters), const_classes + 1)
        self.softmax = nn.LogSoftmax()
        self.init_parameters()
        self.noise  = GaussianNoise(0.01)

    def forward(self, x_src):
        emb_src = self.emb_src(x_src).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        convs_src = [F.relu(conv(emb_src)).squeeze(3) for conv in self.convs_src]  # [batch_size * num_filter * length]
        pools_src = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs_src] # [batch_size * num_filter]
        pred = torch.cat(pools_src, 1)  # batch_size * num_filters_sum
        #pred = self.noise(pred)
        highway = self.highway(pred)
        pred = F.sigmoid(highway) *  F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred = self.softmax(self.lin(self.dropout(pred)))
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
