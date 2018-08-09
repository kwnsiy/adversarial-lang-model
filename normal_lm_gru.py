# -*- coding:utf-8 -*-

import os
import random
import math

import argparse
import tqdm

import pdb

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from seq2seq_iter import GenDataIter, DisDataIter

from lm.normal_generator_gru import Generator
from lm.normal_discriminator import Discriminator

from lm.normal_rollout import Rollout

"""
fixed-cond-lang-model
"""


# ================== Parameter Definition =================
parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
print(opt)

# pretrain 
pretrain = True
model_path = "null"
#pretrain = False
#model_path = "./lm/out/mle_epochs/norm_lm_gru,5,epoch,11.748903274536133,33.72813034057617,.model"

# VOCAB_SIZE
SRC_VOCAB_SIZE = 9000
TGT_VOCAB_SIZE = 9000
padding_id = 3
start_id = 1

# Basic Training Paramters
SEED = 256
BATCH_SIZE    = 64
TOTAL_BATCH   = 100
GENERATED_NUM = 87000
POSITIVE_FILE = ['./data/train_src_words.txt', './data/train_tgt_words.txt', './data/train_tgt_da_id.txt']
EVAL_FILE     = ['./data/test_src_words.txt' , './data/test_tgt_words.txt' , './data/test_tgt_da_id.txt']
NEGATIVE_FILE = ['./lm/out/gen/norm_lm_gru_fake_src_words.txt' , './lm/out/gen/norm_lm_gru_fake_tgt_words.txt' , './lm/out/gen/norm_lm_gru_fake_tgt_da_id.txt']
TEMP_FILE     = ['./lm/out/gen/norm_lm_gru_temp_src_words.txt' , './lm/out/gen/norm_lm_gru_temp_tgt_words.txt' , './lm/out/gen/norm_lm_gru_temp_tgt_da_id.txt']
VOCAB_MODEL   = ['./data/9000.model', './data/9000.model']
PRE_EPOCH_NUM = 8

# gpu
opt.cuda = True

# Genrator Parameters
g_emb_dim = 256    # * 2
g_hidden_dim = 256 * 2# * 4
g_sequence_len = 100

# Discriminator Parameters
d_emb_dim = 128
d_filter_sizes = [2, 5, 7, 10]
d_num_filters = [128, 128, 128, 128]
d_dropout = 0.75
d_num_class = 2

import sentencepiece as spm
sp_src = spm.SentencePieceProcessor()
sp_src.Load(VOCAB_MODEL[0])
sp_dec = spm.SentencePieceProcessor()
sp_dec.Load(VOCAB_MODEL[1])

# generate sample
def generate_samples(model, batch_size, generated_num, output_file, iter):
    samples = []
    das = []
    iter.reset()
    for (src, tgt, da, rf) in tqdm(iter, mininterval=2, desc=' - Training', leave=False):
        da = Variable(da).cuda()
        sample = model.sample(src.size(0), g_sequence_len, da).cpu().data.numpy().tolist()
        samples.extend(sample)
        das.extend([str(x[0]) for x in da.tolist()])
    with open(output_file[0], 'w') as fout:
        for sample in samples:
            string = sp_dec.DecodeIds(sample)
            fout.write('%s\n' % string)    
    with open(output_file[1], 'w') as fout:
        for sample in samples:
            string = sp_dec.DecodeIds(sample)
            fout.write('%s\n' % string)    
    with open(output_file[2], 'w') as fout:
        string = "\n".join(das)
        fout.write('%s\n' % string)    


# train epoch
def train_epoch(model, data_iter, criterion, optimizer):
    data_iter.reset()
    total_loss, total_words, pred, ite, acc = 0., 0., 0., 0., 0
    for (src, tgt, da, rf) in tqdm(data_iter, mininterval=2, desc=' - Training', leave=False):
        src, tgt_sos, tgt_pad, da, rf  = Variable(src), Variable(tgt[0]), Variable(tgt[1]), Variable(da), Variable(rf)
        if opt.cuda: src, tgt_sos, tgt_pad, da, rf = src.cuda(), tgt_sos.cuda(), tgt_pad.cuda(), da.cuda(), rf.cuda()
        tgt_pad = tgt_pad.contiguous().view(-1)
        optimizer.zero_grad()
        pred = model.forward(tgt_sos, da)
        loss = criterion(pred, tgt_pad)
        total_loss += loss.data[0]
        total_words += torch.nonzero(tgt_pad - padding_id).size(0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ite += 1
    return torch.exp(total_loss / total_words)

# eval epoch
def eval_epoch(model, data_iter, criterion):
    data_iter.reset()
    total_loss, total_words, pred, ite, acc = 0., 0., 0., 0., 0
    for (src, tgt, da, rf) in tqdm(data_iter, mininterval=2, desc=' - Training', leave=False):
        src, tgt_sos, tgt_pad, da, rf  = Variable(src), Variable(tgt[0]), Variable(tgt[1]), Variable(da), Variable(rf)
        if opt.cuda: src, tgt_sos, tgt_pad, da, rf = src.cuda(), tgt_sos.cuda(), tgt_pad.cuda(), da.cuda(), rf.cuda()
        tgt_pad = tgt_pad.contiguous().view(-1)
        pred = model.forward(tgt_sos, da)
        loss = criterion(pred, tgt_pad)
        total_loss += loss.data[0]
        total_words += torch.nonzero(tgt_pad - padding_id).size(0)
        ite += 1
    return torch.exp(total_loss / total_words)

def train_disc(model, data_iter, criterion, optimizer):
    data_iter.reset()
    total_loss, total_words, pred, ite, acc = 0., 0., 0., 0., 0
    for (src, tgt, da, rf, drf) in tqdm(data_iter, mininterval=2, desc=' - Training', leave=False):
        drf = rf
        src, tgt_sos, tgt_pad, da, rf, drf  = Variable(src), Variable(tgt[0]), Variable(tgt[1]), Variable(da[:,1]), Variable(rf), Variable(drf)
        if opt.cuda: src, tgt_sos, tgt_pad, da, rf, drf = src.cuda(), tgt_sos.cuda(), tgt_pad.cuda(), da.cuda(), rf.cuda(), drf.cuda()
        optimizer.zero_grad()
        pred = model.forward(src)
        loss = criterion(pred, drf)
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        ite += 1
        _, predicted = torch.max(pred.data, 1)
        acc += (predicted == drf).sum().item() / drf.size(0)
    print (acc/ ite)
    return total_loss / ite, acc/ ite

# eval epoch
def eval_disc(model, data_iter, criterion):
    data_iter.reset()
    total_loss, total_words, pred, ite, acc = 0., 0., 0., 0., 0
    for (src, tgt, da, rf, drf) in tqdm(data_iter, mininterval=2, desc=' - Training', leave=False):
        drf = rf
        src, tgt_sos, tgt_pad, da, rf, drf  = Variable(src), Variable(tgt[0]), Variable(tgt[1]), Variable(da[:,1]), Variable(rf), Variable(drf)
        if opt.cuda: src, tgt_sos, tgt_pad, da, rf, drf = src.cuda(), tgt_sos.cuda(), tgt_pad.cuda(), da.cuda(), rf.cuda(), drf.cuda()
        pred = model.forward(src)
        loss = criterion(pred, drf)
        total_loss += loss.data[0]
        ite += 1
        _, predicted = torch.max(pred.data, 1)
        acc += (predicted == drf).sum().item() / drf.size(0)
    print (acc/ ite)
    return total_loss / ite, acc/ ite


# gan loss
class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda: one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda: one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss = - torch.sum(loss)
        return loss

# main
def main():
    # Seed
    random.seed(SEED)
    np.random.seed(SEED)
        
    # Define Networks
    generator = Generator(TGT_VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda, start_id, padding_id)
    discriminator = Discriminator(d_num_class, 2 - 1, TGT_VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout, padding_id)
        
    # cuda    
    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    
    # generator 
    eval_criterion = nn.NLLLoss(size_average=False, ignore_index=padding_id)
    gen_criterion = nn.NLLLoss(size_average=False)
    gen_optimizer = optim.Adam(generator.parameters(), lr = 1e-3, weight_decay = 1e-6)
    if opt.cuda: gen_criterion = gen_criterion.cuda()

    # Pretrain Generator using MLE
    if pretrain:
        print('Pretrain with MLE ...')
        gen_data_iter  = GenDataIter(POSITIVE_FILE, BATCH_SIZE, padding_id, start_id, VOCAB_MODEL)
        eval_iter      = GenDataIter(EVAL_FILE, BATCH_SIZE, padding_id, start_id, VOCAB_MODEL)
        for epoch in range(PRE_EPOCH_NUM):
            loss       = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
            print('Epoch [%d] Model Loss: %f'% (epoch, loss))
            train_loss = eval_epoch(generator, gen_data_iter, eval_criterion )   
            print('Epoch [%d] Model* Loss: %f'% (epoch, train_loss))
            eval_loss  = eval_epoch(generator, eval_iter, eval_criterion )      
            print('Epoch [%d] True Loss: %f' % (epoch, eval_loss))
            torch.save(generator.state_dict(), "./lm/out/mle_epochs/norm_lm_gru," + str(epoch) + ",epoch," + 
                str(float(train_loss)) + "," + str(float(eval_loss))+ ",.model")
    else: 
        generator.load_state_dict(torch.load(model_path))
        generator = generator.cuda()
    
    # Pretrain Discriminator
    dis_criterion = nn.NLLLoss(size_average=False)
    dis_optimizer = optim.SGD(discriminator.parameters(), lr=1e-3)
    if opt.cuda: dis_criterion = dis_criterion.cuda()
    print('Pretrain Dsicriminator ...')
    disc_loss = 0
    for epoch in range(3):
        neg_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE * 10, padding_id, start_id, VOCAB_MODEL)
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE, neg_iter)
        dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE, VOCAB_MODEL, g_sequence_len)
        for _ in range(3):
            disc_loss, disc_acc = train_disc(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
            print('Epoch [%d], loss, acc: %f, %f' % (epoch, disc_loss, disc_acc))
    
    # eval dicriminator
    neg_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE * 10, padding_id, start_id, VOCAB_MODEL, sample_per_iter=1)
    generate_samples(generator, BATCH_SIZE*10, 1, TEMP_FILE, neg_iter)
    dis_data_iter = DisDataIter(POSITIVE_FILE, TEMP_FILE, BATCH_SIZE, VOCAB_MODEL, g_sequence_len)
    train_loss, train_acc = eval_disc(discriminator, dis_data_iter, dis_criterion)
    dis_data_iter = DisDataIter(EVAL_FILE, TEMP_FILE, BATCH_SIZE, VOCAB_MODEL, g_sequence_len)
    eval_loss, eval_acc = eval_disc(discriminator, dis_data_iter, dis_criterion)
    
    # eval pretrain discriminators
    torch.save(discriminator.state_dict(), "./lm/out/mle_epochs/norm_lm_disc_pre," +
    	str(float(disc_loss)) + "," + str(float(disc_acc)) + "," + 
    	str(float(train_loss)) + "," + str(float(train_acc)) + "," + 
    	str(float(eval_loss)) + "," + str(float(eval_acc)) + ",.model")
    print('pre-train-disc, train_loss, train_acc: %f, %f' % (train_loss, train_acc))
    print('pre-train-disc, eval_loss, eval_acc: %f, %f' % (train_loss, eval_acc))

    # Adversarial Training 
    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters(), lr=1e-5, weight_decay=0.0) # @1e-5
    if opt.cuda: gen_gan_loss = gen_gan_loss.cuda()
    dis_criterion = nn.NLLLoss(size_average=False)
    dis_optimizer = optim.SGD(discriminator.parameters(), lr=1e-3) # @1e-3
    if opt.cuda: dis_criterion = dis_criterion.cuda()

    rollout = Rollout(generator, 1.0) # 謎処理（やると不思議に動く: 論文には載っていない） 0.8
    gan_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE, padding_id, start_id, VOCAB_MODEL)
    print('#####################################################')
    print('Start Adeversatial Training...\n')
    for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step
        gan_iter.reset(); i = 0
        for src, tgt, da, rf  in gan_iter:
            if i == 20: break
            da  = da.cuda()
            samples = generator.sample(da.size(0), da.size(1), da)
            # construct the input to the genrator, add zeros before samples and delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor) + start_id
            if samples.is_cuda: zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view((-1,))
            # calculate the reward
            rewards = rollout.get_reward(inputs, 4, discriminator, Variable(da).cuda()) # @4
            rewards = Variable(torch.Tensor(rewards))
            if opt.cuda: rewards = torch.exp(rewards.cuda()).contiguous().view((-1,))
            prob = generator.forward(inputs, da)
            loss = gen_gan_loss(prob, targets, rewards)
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()
            print ("Batch [%d] GAN-loss: %f" % (total_batch, loss))
            # if random.random() < 1.1: # teacher-forsing 
            if (float(loss) > 500.0) or (random.random() < 0.5): # teacher-forsing: 十分に学習したら教師強制しない
              prob = generator.forward(Variable(tgt[0]).cuda(), Variable(da).cuda())
              loss = gen_gan_loss(prob, Variable(tgt[1]).contiguous().view((-1,)).cuda(), rewards*0.0 + 1.0)
              gen_gan_optm.zero_grad()
              loss.backward()
              gen_gan_optm.step()
              print ("Batch [%d] GAN-loss (forsing): %f" % (total_batch, loss))
            i += 1
        rollout.update_params()
        
        # eval generator
        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:                
            eval_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE, padding_id, start_id, VOCAB_MODEL)
            train_loss = eval_epoch(generator, eval_iter, eval_criterion)
            print('Batch [%d] Model Loss: %f' % (total_batch, train_loss))
            eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE, padding_id, start_id, VOCAB_MODEL)
            eval_loss = eval_epoch(generator, eval_iter, eval_criterion )
            print('Batch [%d] True Loss: %f' % (total_batch, eval_loss))
            torch.save(generator.state_dict(), "./lm/out/gan_epochs/norm_lm," + str(total_batch) + 
                ",batch," + str(float(train_loss)) + "," + str(float(eval_loss)) + ".model")
        
        # update discriminator
        disc_loss = 0
        base = 1
        k = 3
        for _ in range(base): # 2
            neg_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE * 10, padding_id, start_id, VOCAB_MODEL)
            generate_samples(generator, BATCH_SIZE*10, GENERATED_NUM, NEGATIVE_FILE, neg_iter)
            dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE, VOCAB_MODEL, g_sequence_len, sample_per_iter=10000) #@10000
            for _ in range(k): # 2
                disc_loss, disc_acc = train_disc(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
            print('Batch [%d] Discriminator Loss: %f, %f' % (total_batch, disc_loss, disc_acc))

        # eval dicriminator
        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1: 
            neg_iter = GenDataIter(POSITIVE_FILE, 1, padding_id, start_id, VOCAB_MODEL, sample_per_iter=1)
            generate_samples(generator, BATCH_SIZE*10, 1, TEMP_FILE, neg_iter)
            dis_data_iter = DisDataIter(POSITIVE_FILE, TEMP_FILE, BATCH_SIZE*10, VOCAB_MODEL, g_sequence_len)
            train_loss, train_acc = eval_disc(discriminator, dis_data_iter, dis_criterion)
            dis_data_iter = DisDataIter(EVAL_FILE, TEMP_FILE, BATCH_SIZE*10, VOCAB_MODEL, g_sequence_len)
            eval_loss, eval_acc = eval_disc(discriminator, dis_data_iter, dis_criterion)
            # eval pretrain discriminators
            torch.save(discriminator.state_dict(), "./lm/out/gan_epochs/norm_lm_disc," + str(total_batch) + ",batch," + 
                str(float(disc_loss))  + ","  + str(float(disc_acc)) + "," +  
                str(float(train_loss)) + ","  + str(float(train_acc)) + "," + 
                str(float(eval_loss))  + ","  + str(float(eval_acc))  + "," + ".model")
    

if __name__ == '__main__':
    main()
