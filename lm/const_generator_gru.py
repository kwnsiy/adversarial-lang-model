# -*- coding: utf-8 -*-

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pdb

# gauss noise
class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev
    def forward(self, din):
        if self.training:
          return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din

# generator
class Generator(nn.Module):
    """Generator """
    def __init__(self, num_emb, const_dim, emb_dim, hidden_dim, use_cuda, start_idx, padding_idx):
        super(Generator, self).__init__()
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.emb = nn.Embedding(num_emb, emb_dim, padding_idx=padding_idx)
        self.emb_da = nn.Embedding(const_dim + 1, 50)
        self.lstm = nn.GRU(emb_dim + 50, hidden_dim, batch_first=True, dropout=0.0)
        self.lin = nn.Linear(hidden_dim, num_emb)
        self.softmax = nn.LogSoftmax()
        self.init_params()
        self.start_idx = start_idx
        #self.noise = GaussianNoise(0.1)

    def forward(self, x, x_da):
        emb_x    = self.emb(x)
        emb_da   = self.emb_da(x_da)
        emb_conc = torch.cat((emb_x, emb_da), dim = 2)
        h0, c0 = self.init_hidden(x.size(0))
        output, h = self.lstm(emb_conc, h0)
        pred = self.softmax(self.lin(output.contiguous().view(-1, self.hidden_dim)))
        return pred

    def step(self, x, h, x_da, dim=2):
        emb_x = self.emb(x)
        emb_da = self.emb_da(x_da)
        emb_conc = torch.cat((emb_x, emb_da), dim = dim)
        output, h = self.lstm(emb_conc, h)
        pred = F.softmax(self.lin(output.view(-1, self.hidden_dim)))
        return pred, h

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        if self.use_cuda: h, c = h.cuda(), c.cuda()
        return h, c
    
    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def sample(self, batch_size, seq_len, x_da, x=None):
        res = []
        flag = False # whether sample from zero
        if x is None: flag = True
        if flag: x  = Variable(torch.ones((batch_size, 1)).long())
        if self.use_cuda:
            x = x.cuda()
            x_da = x_da.cuda()
        h, c = self.init_hidden(batch_size)
        samples = []
        if flag:
            x_da = x_da[:, 1].view(-1,1)
            for i in range(seq_len):
                output, h = self.step(x, h, x_da, dim=2)
                x = output.multinomial(1)
                samples.append(x)
        else:
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            lis_da = x_da.chunk(x_da.size(1), dim=1)
            x_da_ = x_da[:, 0].view(-1,1)
            for i in range(given_len):
                output, h = self.step(lis[i], h, lis_da[i], dim=2)
                samples.append(lis[i])
            x = output.multinomial(1)
            for i in range(given_len, seq_len):
                samples.append(x)
                output, h = self.step(x, h, x_da_, dim=2)
                x = output.multinomial(1)
        output = torch.cat(samples, dim=1)
        return output
