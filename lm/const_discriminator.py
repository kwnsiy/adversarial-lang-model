# -*- coding: utf-8 -*-

import os
import random
import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb 
class Discriminator(nn.Module):

    def __init__(self, num_classes, const_classes, vocab_size, emb_dim, filter_sizes, num_filters, dropout, padding_idx):
        super(Discriminator, self).__init__()
        self.emb    = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.da_emb = nn.Embedding(const_classes + 1, emb_dim)
        self.convs  = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_dim*2)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters)) # *
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(sum(num_filters), num_classes)
        self.softmax = nn.LogSoftmax() # LogSoftmax
        self.init_parameters()
    
    def forward(self, x, temp):
        emb_da   = self.da_emb(temp).unsqueeze(1)
        emb   = self.emb(x).unsqueeze(1)
        emb_c = torch.cat((emb, emb_da), dim = 3)
        convs = [F.relu(conv(emb_c)).squeeze(3) for conv in self.convs] 
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] 
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = F.sigmoid(highway) *  F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred = self.softmax(self.lin(self.dropout(pred)))
        return pred
        
    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
