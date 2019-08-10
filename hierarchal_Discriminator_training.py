#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:10:45 2019

@author: r17935avinash
"""

################################ IMPORT LIBRARIES ###############################################################
import torch
import numpy as np
import pykp.io
import torch.nn as nn
from utils.statistics import RewardStatistics
from utils.time_log import time_since
import time
from sequence_generator import SequenceGenerator
from utils.report import export_train_and_valid_loss, export_train_and_valid_reward
import sys
import logging
import os
from evaluate import evaluate_reward
from pykp.reward import *
import math
EPS = 1e-8
import argparse
import config
import logging
import os
import json
from pykp.io import KeyphraseDataset
from pykp.model import Seq2SeqModel
from torch.optim import Adam
import pykp
from pykp.model import Seq2SeqModel
import train_ml
import train_rl

from utils.time_log import time_since
from utils.data_loader import load_data_and_vocab
from utils.string_helper import convert_list_to_kphs
import time
import numpy as np
import random
from torch import device 
from hierarchal_Discriminator import Discriminator
from torch.nn import functional as F
#####################################################################################################
opt = argparse.Namespace(attn_mode='concat', baseline='self', batch_size=32, batch_workers=4, bidirectional=True, bridge='copy', checkpoint_interval=4000, copy_attention=True, copy_input_feeding=False, coverage_attn=False, coverage_loss=False, custom_data_filename_suffix=False, custom_vocab_filename_suffix=False, data='data/kp20k_tg_sorted/', data_filename_suffix='', dec_layers=1, decay_method='', decoder_size=300, decoder_type='rnn', delimiter_type=0, delimiter_word='<sep>', device=device(type='cuda', index=3
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ), disable_early_stop_rl=False, dropout=0.1, dynamic_dict=True, early_stop_tolerance=4, enc_layers=1, encoder_size=150, encoder_type='rnn', epochs=20, exp='kp20k.rl.one2many.cat.copy.bi-directional', exp_path='exp/kp20k.rl.one2many.cat.copy.bi-directional.20190701-192604', final_perturb_std=0, fix_word_vecs_dec=False, fix_word_vecs_enc=False, goal_vector_mode=0, goal_vector_size=16, gpuid=3, init_perturb_std=0, input_feeding=False, lambda_coverage=1, lambda_orthogonal=0.03, lambda_target_encoder=0.03, learning_rate=0.001, learning_rate_decay=0.5, learning_rate_decay_rl=False, learning_rate_rl=5e-05, loss_normalization='tokens', manager_mode=1, match_type='exact', max_grad_norm=1, max_length=60, max_sample_length=6, max_unk_words=1000, mc_rollouts=False, model_path='model/kp20k.rl.one2many.cat.copy.bi-directional.20190701-192604', must_teacher_forcing=False, num_predictions=1, num_rollouts=3, one2many=True, one2many_mode=1, optim='adam', orthogonal_loss=False, param_init=0.1, perturb_baseline=False, perturb_decay_factor=0.0001, perturb_decay_mode=1, pre_word_vecs_dec=None, pre_word_vecs_enc=None, pretrained_model='model/kp20k.ml.one2many.cat.copy.bi-directional.20190628-114655/kp20k.ml.one2many.cat.copy.bi-directional.epoch=2.batch=54573.total_batch=116000.model', regularization_factor=0.0, regularization_type=0, remove_src_eos=False, replace_unk=True, report_every=10, review_attn=False, reward_shaping=False, reward_type=7, save_model='model', scheduled_sampling=False, scheduled_sampling_batches=10000, seed=9527, separate_present_absent=True, share_embeddings=True, source_representation_queue_size=128, source_representation_sample_size=32, start_checkpoint_at=2, start_decay_at=8, start_epoch=1, target_encoder_size=64, teacher_forcing_ratio=0, timemark='20190701-192604', title_guided=False, topk='G', train_from='', train_ml=False, train_rl=True, truncated_decoder=0, use_target_encoder=False, vocab='data/kp20k_separated/', vocab_filename_suffix='', vocab_size=100002, warmup_steps=4000, word_vec_size=100, words_min_frequency=0)
devices = "cuda:3"
            
##### TUNE HYPERPARAMETERS ##############
hidden_dim = 150
embedding_dim = 200
n_layers = 2 
##  batch_reward_stat, log_selected_token_dist = train_one_batch(batch, generator, optimizer_rl, opt, perturb_std)
#########################################################
def train_one_batch(D_model,one2many_batch, generator, opt,perturb_std):
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, title, title_oov, title_lens, title_mask = one2many_batch
    one2many = opt.one2many
    one2many_mode = opt.one2many_mode
    if one2many and one2many_mode > 1:
        num_predictions = opt.num_predictions
    else:
        num_predictions = 1
    

    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
#    src = src.cuda()
#    src_mask = src_mask.cuda()
#    src_oov = src_oov.cuda()
    # trg = trg.to(opt.device)
    # trg_mask = trg_mask.to(opt.device)
    # trg_oov = trg_oov.to(opt.device)

    if opt.title_guided:
        title = title.to(opt.device)
        title_mask = title_mask.to(opt.device)
        
    eos_idx = opt.word2idx[pykp.io.EOS_WORD]
    delimiter_word = opt.delimiter_word
    batch_size = src.size(0)
    topk = opt.topk
    reward_type = opt.reward_type
    reward_shaping = opt.reward_shaping
    baseline = opt.baseline
    match_type = opt.match_type
    regularization_type = opt.regularization_type ## DNT
    regularization_factor = opt.regularization_factor ##DNT
    
    if regularization_type == 2:
        entropy_regularize = True
    else:
        entropy_regularize = False
    
    start_time = time.time()
    sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask, entropy, location_of_eos_for_each_batch, location_of_peos_for_each_batch = generator.sample(
        src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=False, one2many=one2many,
        one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=perturb_std, entropy_regularize=entropy_regularize, title=title, title_lens=title_lens, title_mask=title_mask)
    
    pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx, delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                              src_str_list, opt.separate_present_absent, pykp.io.PEOS_WORD)
    target_str_2dlist = convert_list_to_kphs(trg)
    """
     src = [batch_size,abstract_seq_len]
     target_str_2dlist = list of list of true keyphrases
     pred_str_2dlist = list of list of false keyphrases
    
    """
    total_abstract_loss = 0
    batch_mine = 0
    abstract_t = torch.Tensor([]).to(devices)
    abstract_f = torch.Tensor([]).to(devices)
    kph_t = torch.Tensor([]).to(devices)
    kph_f = torch.Tensor([]).to(devices)    
    h_kph_t_size = 0
    h_kph_f_size = 0 
    len_list_t = []
    len_list_f = []
    for idx, (src_list, pred_str_list,target_str_list) in enumerate(zip(src, pred_str_2dlist,target_str_2dlist)):
        
        batch_mine+=1
        if (len(target_str_list)==0 or len(pred_str_list)==0):
            continue
            
        h_abstract_t,h_kph_t = D_model.get_hidden_states(src_list,target_str_list)
        h_abstract_f,h_kph_f = D_model.get_hidden_states(src_list,pred_str_list)
        h_kph_t_size = max(h_kph_t_size,h_kph_t.size(1))
        h_kph_f_size = max(h_kph_f_size,h_kph_f.size(1))
  #  print(h_kph_t_size,h_kph_f_size)  
    
    for idx, (src_list, pred_str_list,target_str_list) in enumerate(zip(src, pred_str_2dlist,target_str_2dlist)):
        
        batch_mine+=1
        if (len(target_str_list)==0 or len(pred_str_list)==0):
            continue  
        h_abstract_t,h_kph_t = D_model.get_hidden_states(src_list,target_str_list)
        h_abstract_f,h_kph_f = D_model.get_hidden_states(src_list,pred_str_list)
     #   print(h_abstract_t.size(),h_kph_t.size())
        p1d = (0,0,0,h_kph_t_size - h_kph_t.size(1))
        p2d = (0,0,0,h_kph_f_size - h_kph_f.size(1))
   #     print("sfgs:",h_kph_t_size - h_kph_t.size(1),h_kph_f_size - h_kph_f.size(1))
        h_kph_t = F.pad(h_kph_t,p1d)
        h_kph_f = F.pad(h_kph_f,p2d)
  
        abstract_t = torch.cat((abstract_t,h_abstract_t),dim=0)
        abstract_f = torch.cat((abstract_f,h_abstract_f),dim=0)
        kph_t = torch.cat((kph_t,h_kph_t),dim=0)
        kph_f = torch.cat((kph_f,h_kph_f),dim=0)
   
       
    posit = torch.cat((abstract_t,kph_t),dim=1)
    negit = torch.cat((abstract_f,kph_f),dim=1)
   # print(posit,negit)
    real_rewards,abstract_loss_real = D_model.calc_keyphrase_score(posit,1)
    
    fake_rewards,abstract_loss_fake = D_model.calc_score_loss(negit,0)
    avg_batch_loss = abstract_loss_real + abstract_loss_fake
    avg_real = real_rewards
    avg_fake = fake_rewards
    return avg_batch_loss,avg_real,avg_fake

def main():
    #print("agsnf efnghrrqthg")
    clip = 5
    start_time = time.time()
    train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_data_and_vocab(opt, load_train=True)
    load_data_time = time_since(start_time)
    logging.info('Time for loading the data: %.1f' % load_data_time)
    
    model = Seq2SeqModel(opt)
    #model = model.device()
    #print("The Device is",opt.gpuid)
    #model = model.to(devices)
    model = model.to(devices)
    
   # model.load_state_dict(torch.load("model/kp20k.ml.one2many.cat.copy.bi-directional.20190628-114655/kp20k.ml.one2many.cat.copy.bi-directional.epoch=2.batch=54573.total_batch=116000.model"))
    model.load_state_dict(torch.load("model/kp20k.ml.one2many.cat.copy.bi-directional.20190715-132016/kp20k.ml.one2many.cat.copy.bi-directional.epoch=3.batch=26098.total_batch=108000.model"))
    generator = SequenceGenerator(model,
                                  bos_idx=opt.word2idx[pykp.io.BOS_WORD],
                                  eos_idx=opt.word2idx[pykp.io.EOS_WORD],
                                  pad_idx=opt.word2idx[pykp.io.PAD_WORD],
                                  peos_idx=opt.word2idx[pykp.io.PEOS_WORD],
                                  beam_size=1,
                                  max_sequence_length=opt.max_length,
                                  copy_attn=opt.copy_attention,
                                  coverage_attn=opt.coverage_attn,
                                  review_attn=opt.review_attn,
                                  cuda=opt.gpuid > -1
                                  )
    
    init_perturb_std = opt.init_perturb_std
    final_perturb_std = opt.final_perturb_std
    perturb_decay_factor = opt.perturb_decay_factor
    perturb_decay_mode = opt.perturb_decay_mode
    
    D_model = Discriminator(opt.vocab_size,embedding_dim,hidden_dim,n_layers,opt.word2idx[pykp.io.PAD_WORD])
    
    print("The Discriminator statistics are ",D_model)
    
    if torch.cuda.is_available():
        D_model = D_model.to(devices)
    
    D_model.train()
    
    D_optimizer = torch.optim.Adam(D_model.parameters(),lr=0.001)
    
    print("gdsf")
    total_epochs = 5
    for epoch in range(total_epochs):
        
        total_batch = 0
        print("Starting with epoch:",epoch)
        for batch_i, batch in enumerate(train_data_loader):
            total_batch+=1
            D_optimizer.zero_grad()
            
            if perturb_decay_mode == 0:  # do not decay
                perturb_std = init_perturb_std
            elif perturb_decay_mode == 1:  # exponential decay
                perturb_std = final_perturb_std + (init_perturb_std - final_perturb_std) * math.exp(-1. * total_batch * perturb_decay_factor)
            elif perturb_decay_mode == 2:  # steps decay
                perturb_std = init_perturb_std * math.pow(perturb_decay_factor, math.floor((1+total_batch)/4000))


            avg_batch_loss,real_r,fake_r = train_one_batch(D_model,batch,generator,opt,perturb_std)
#            print("Currently loss is",avg_batch_loss.item())
#            print("Currently real loss is",real_r.item())
#            print("Currently fake loss is",fake_r.item())
#            state_dfs = D_model.state_dict()
#            torch.save(state_dfs,"Checkpoint_" + str(epoch) + ".pth.tar")
#
            
            
            if batch_i%350==0:
                print("Currently loss is",avg_batch_loss.item())
                print("Currently real loss is",real_r.item())
                print("Currently fake loss is",fake_r.item())
                
                print("Saving the file ...............----------->>>>>")        
                state_dfs = D_model.state_dict()
                torch.save(state_dfs,"Discriminator_checkpts/hierarchal_Dis_" + str(epoch) + ".pth.tar")

            
            
            torch.nn.utils.clip_grad_norm_(D_model.parameters(), clip)
            avg_batch_loss.backward()
            D_optimizer.step()
            
            #sys.exit()
        
        print("Saving the file ...............----------->>>>>")        
        state_dfs = D_model.state_dict()
        torch.save(state_dfs,"Discriminator_checkpts/hierarchal_Dis_" + str(epoch) + ".pth.tar")

######################################            
            
        
main()
    
    
    
