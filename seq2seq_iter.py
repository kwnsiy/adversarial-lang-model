# -*- coding:utf-8 -*-

import os
import random
import math
import tqdm
import numpy as np
import torch
import pdb
import numpy as np 
import sentencepiece as spm 

# coding: utf-8 
from itertools import chain, repeat, islice

def pad_infinite(iterable, padding=None): return chain(iterable, repeat(padding))
def pad(iterable, size, padding=None): return islice(pad_infinite(iterable, padding), size)

class GenDataIter(object):
    def __init__(self, data_file, batch_size, padding_idx, start_idx, vocab_model, sample_per_iter=10000000000000):
        super(GenDataIter, self).__init__()
        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.vocab_model = vocab_model
        self.batch_size = batch_size
        self.data_src, self.data_tgt, self.data_da = self.read_file(data_file, self.vocab_model)
        self.data_num = len(self.data_tgt)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0
        self.sample_per_iter = sample_per_iter

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        c = list(zip(self.data_src, self.data_tgt, self.data_da))
        random.shuffle(c)
        self.data_src, self.data_tgt, self.data_da = zip(*c)

    def next(self):
        if self.idx >= self.data_num: raise StopIteration
        if self.idx >= self.sample_per_iter: raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        batch_size_ = len(index)
        d_src  = [self.data_src[i] for i in index] # src
        d_tgt = [self.data_tgt[i] for i in index] # src
        d_da = [self.data_da[i] for i in index] # src
        maxlen = max([max(map(len, d_src)), max(map(len, d_tgt))]) # max-length of minibatch
        # lef padding - src
        d_src = [np.pad(src, (maxlen - len(src) + 2, 0), 'constant', constant_values=3) for src in d_src]
        d_src = torch.LongTensor(np.asarray(d_src, dtype='int64'))
        # right 
        d_tgt = [np.pad(tgt, (0, maxlen - len(tgt) + 1), 'constant', constant_values=3) for tgt in d_tgt]
        d_tgt = torch.LongTensor(np.asarray(d_tgt, dtype='int64'))
        d_tgt_sos   =  torch.cat([torch.zeros(batch_size_, 1).long() + self.start_idx, d_tgt], dim=1)
        d_tgt_pad   =  torch.cat([d_tgt, torch.zeros(batch_size_, 1).long() + self.padding_idx], dim=1)
        # da 
        da     = torch.LongTensor(np.asarray([[_] * (maxlen + 2) for _ in d_da], dtype='int64'))
        self.idx += self.batch_size
        return d_src, [d_tgt_sos, d_tgt_pad], da, da  
        """
        sp_src = spm.SentencePieceProcessor()
        sp_src.Load(self.vocab_model[0])
        sp_dec = spm.SentencePieceProcessor()
        sp_dec.Load(self.vocab_model[1])
        sp_dec.DecodeIds(self.data_lis_dec[0].tolist())
        """

    def read_file(self, data_file, vocab_model):
        sp_src = spm.SentencePieceProcessor()
        sp_src.Load(vocab_model[0])
        sp_dec = spm.SentencePieceProcessor()
        sp_dec.Load(vocab_model[1])
        with open(data_file[0], 'r') as f :  lines = f.readlines()
        with open(data_file[1], 'r') as f2:  lines2 = f2.readlines()
        with open(data_file[2], 'r') as f3:  lines3 = f3.readlines()
        lis_src, lis_tgt, lis_da = [], [], []
        for line in lines: 
            lis_src.append(sp_src.EncodeAsIds(line.strip()))
        for line in lines2: 
            lis_tgt.append(sp_src.EncodeAsIds(line.strip()) + [2])
        for line in lines3:
            lis_da.append(int(line.strip()))
        return lis_src, lis_tgt, lis_da

"""
        sp_src = spm.SentencePieceProcessor()
        sp_src.Load(vocab_model[0])
        sp_dec = spm.SentencePieceProcessor()
        sp_dec.Load(vocab_model[1])
        for line in lines: 
            line = sp_src.EncodeAsIds(line.strip()) 
            l = len(line)
            pad_line = np.pad(line, (maxlen - l, 0), 'constant', constant_values=3)
            lis_src.append(pad_line)
        for line2 in lines2: 
            line2 = sp_dec.EncodeAsIds(line2.strip()) 
            l = len(line2)
            pad_line2 = np.pad(line2 + [2], (0, maxlen - l + 1), 'constant', constant_values=3)
            lis_dec.append(pad_line2)
        return lis_src, lis_dec
"""


class DisDataIter(object):
    def __init__(self, real_data_file, fake_data_file, batch_size, vocab_model, seq_maxlen, sample_per_iter=100000000000):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        real_data_src, real_data_tgt, real_data_da = self.read_file(real_data_file, vocab_model)
        fake_data_src, fake_data_tgt, fake_data_da = self.read_file(fake_data_file, vocab_model)
        self.src                    = real_data_src + fake_data_src
        self.tgt                    = real_data_tgt + fake_data_tgt
        self.da                     = real_data_da + fake_data_da
        self.rf                     = [1 for _ in real_data_src] + [0 for _ in fake_data_src]
        self.drf                    = real_data_da + [0 for _ in fake_data_src]
        self.pairs = list(zip(self.src, self.tgt, self.da, self.rf, self.drf))
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0
        self.sample_per_iter = sample_per_iter
        self.seq_maxlen = seq_maxlen

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        #pdb.set_trace()
        if self.idx >= self.data_num:
            raise StopIteration
        if self.idx >= self.sample_per_iter:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        pairs =  [self.pairs[i] for i in index]
        d_src   =  [p[0] for p in pairs]
        d_tgt   =  [p[1] for p in pairs]
        d_da    =  [p[2] for p in pairs]
        d_rf    =  [p[3] for p in pairs]
        d_drf   =  [p[4] for p in pairs]
        
        src = [np.pad(src, (0, 280 - len(src)), 'constant', constant_values=3)[0:self.seq_maxlen] for src in d_src] 
        tgt = [np.pad(tgt, (0, 280 - len(tgt)), 'constant', constant_values=3)[0:self.seq_maxlen] for tgt in d_tgt]
        src = torch.LongTensor(np.asarray(src, dtype='int64'))
        tgt = torch.LongTensor(np.asarray(tgt, dtype='int64'))
        da  = torch.LongTensor(np.asarray([[_] * self.seq_maxlen for _ in d_da], dtype='int64'))
        rf = torch.LongTensor(np.asarray(d_rf, dtype='int64')) 
        drf = torch.LongTensor(np.asarray(d_drf, dtype='int64'))
        
        """
        maxlen = max([max(map(len, d_src)), max(map(len, d_tgt))])
        src = [np.pad(src, (0, maxlen - len(src)), 'constant', constant_values=3) for src in d_src] 
        tgt = [np.pad(tgt, (0, maxlen - len(tgt)), 'constant', constant_values=3) for tgt in d_tgt]
        src = torch.LongTensor(np.asarray(src, dtype='int64'))
        tgt = torch.LongTensor(np.asarray(tgt, dtype='int64'))
        da  = torch.LongTensor(np.asarray([[_] * maxlen for _ in d_da], dtype='int64'))
        rf = torch.LongTensor(np.asarray(d_rf, dtype='int64'))    
        drf = torch.LongTensor(np.asarray(d_drf, dtype='int64'))
        """
        self.idx += self.batch_size
        return src, tgt, da, rf, drf

    def read_file(self, data_file, vocab_model):
        sp_src = spm.SentencePieceProcessor()
        sp_src.Load(vocab_model[0])
        sp_dec = spm.SentencePieceProcessor()
        sp_dec.Load(vocab_model[1])
        with open(data_file[0], 'r') as f :  lines = f.readlines()
        with open(data_file[1], 'r') as f2:  lines2 = f2.readlines()
        with open(data_file[2], 'r') as f3:  lines3 = f3.readlines()
        lis_src, lis_tgt, lis_da = [], [], []
        for line in lines: 
            lis_src.append([1] + sp_src.EncodeAsIds(line.strip()) + [2])
        for line in lines2: 
            lis_tgt.append([1] + sp_src.EncodeAsIds(line.strip()) + [2])
        for line in lines3:
            lis_da.append(int(line.strip()))
        return lis_src, lis_tgt, lis_da
