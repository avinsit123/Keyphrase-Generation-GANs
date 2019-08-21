#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:29:10 2019

@author: r17935avinash
"""
#####################################################################################################################################################################################################################################################################################################################################################

import torch
from sequence_generator import SequenceGenerator
import config
import argparse
from preprocess import read_tokenized_src_file,read_tokenized_trg_file
from utils.data_loader import load_vocab
from pykp.io import build_interactive_predict_dataset, KeyphraseDataset
from torch.utils.data import DataLoader
import predict
import os
from torch import device
import sys
from hierarchal_attention_Discriminator import Discriminator
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
from torch.optim import Adam
import pykp
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
############################################################################################################################

############ DO NOT MODIFY AT ANY COST ########################################
hidden_dim = 150
embedding_dim = 200
n_layers = 2 
devices = "cuda:3"
batch_size = 32
###############################################################

#####################################################################################################
opt = argparse.Namespace(attn_mode='concat', baseline='self', batch_size=32, batch_workers=4, bidirectional=True, bridge='copy', checkpoint_interval=4000, copy_attention=True, copy_input_feeding=False, coverage_attn=False, coverage_loss=False, custom_data_filename_suffix=False, custom_vocab_filename_suffix=False, data='data/kp20k_tg_sorted', data_filename_suffix='', dec_layers=1, decay_method='', decoder_size=300, decoder_type='rnn', delimiter_type=0, delimiter_word='<sep>', device=device(type='cuda', index=2), disable_early_stop_rl=False, dropout=0.1, dynamic_dict=True, early_stop_tolerance=4, enc_layers=1, encoder_size=150, encoder_type='rnn', epochs=20, exp='kp20k.rl.one2many.cat.copy.bi-directional', exp_path='exp/kp20k.rl.one2many.cat.copy.bi-directional.20190701-192604', final_perturb_std=0, fix_word_vecs_dec=False, fix_word_vecs_enc=False, goal_vector_mode=0, goal_vector_size=16, gpuid=1, init_perturb_std=0, input_feeding=False, lambda_coverage=1, lambda_orthogonal=0.03, lambda_target_encoder=0.03, learning_rate=0.001, learning_rate_decay=0.5, learning_rate_decay_rl=False, learning_rate_rl=5e-05, loss_normalization='tokens', manager_mode=1, match_type='exact', max_grad_norm=1, max_length=60, max_sample_length=6, max_unk_words=1000, mc_rollouts=False, model_path='model/kp20k.rl.one2many.cat.copy.bi-directional.20190701-192604', must_teacher_forcing=False, num_predictions=1, num_rollouts=3, one2many=True, one2many_mode=1, optim='adam', orthogonal_loss=False, param_init=0.1, perturb_baseline=False, perturb_decay_factor=0.0001, perturb_decay_mode=1, pre_word_vecs_dec=None, pre_word_vecs_enc=None, pretrained_model='model/kp20k.ml.one2many.cat.copy.bi-directional.20190628-114655/kp20k.ml.one2many.cat.copy.bi-directional.epoch=2.batch=54573.total_batch=116000.model', regularization_factor=0.0, regularization_type=0, remove_src_eos=False, replace_unk=True, report_every=10, review_attn=False, reward_shaping=False, reward_type=7, save_model='model', scheduled_sampling=False, scheduled_sampling_batches=10000, seed=9527, separate_present_absent=True, share_embeddings=True, source_representation_queue_size=128, source_representation_sample_size=32, start_checkpoint_at=2, start_decay_at=8, start_epoch=1, target_encoder_size=64, teacher_forcing_ratio=0, timemark='20190701-192604', title_guided=False, topk='G', train_from='', train_ml=False, train_rl=True, truncated_decoder=0, use_target_encoder=False, vocab='data/kp20k_separated/', vocab_filename_suffix='', vocab_size=100002, warmup_steps=4000, word_vec_size=100, words_min_frequency=0)

#########
class NullWriter(object):
    def write(self, arg):
        pass
    

            
src_file = "data/cross_domain_sorted/word_inspec_testing_context.txt" 
trg_file = "data/cross_domain_sorted/word_inspec_testing_allkeywords.txt"
devices = "cuda:3"

def convert_to_string_list(kph,word2idx):
    s = []
    for kps in kph:
        if kps not in word2idx.keys():
            kps = "<unk>"
        s.append(word2idx[kps])
    return s

def hierarchal_attention_Discriminator(src,pred_str_2dlist):
    D_model = Discriminator(opt.vocab_size,embedding_dim,hidden_dim,n_layers,opt.word2idx[pykp.io.PAD_WORD])
    D_model.load_state_dict(torch.load("Discriminator_checkpts/hierarchal_attention_Dis_4.pth.tar"))
    D_model = D_model.to(devices)
    abstract = torch.Tensor([]).to(devices)
    kph = torch.Tensor([]).to(devices) 
    h_kph_t_size = 0
    h_kph_f_size = 0 
    len_list_t,len_list_f = [],[]
    
    for idx,(src_list,pred_str_list) in enumerate(zip(src,pred_str_2dlist)):
        h_abstract , h_kph = D_model.get_hidden_limit_length(src_list,pred_str_list,70)
        sys.exit()
        len_list_f.append(h_kph_f.size(1))
        h_kph_f_size = max(h_kph_f_size,h_kph_f.size(1)) 
        break

    pred_str_new2dlist = []
    for idx, (src_list, pred_str_list,target_str_list) in enumerate(zip(src, pred_str_2dlist,target_str_2dlist)):
        batch_mine+=1
        if (len(target_str_list)==0 or len(pred_str_list)==0):
            continue  
        pred_str_new2dlist.append(pred_str_list)
        log_selected_token_total_dist = torch.cat((log_selected_token_total_dist,log_selected_token_dist[idx]),dim=0)
        h_abstract_f,h_kph_f = D_model.get_hidden_states(src_list,pred_str_list)
        p2d = (0,0,0,h_kph_f_size - h_kph_f.size(1))
        h_kph_f = F.pad(h_kph_f,p2d)
        abstract_f = torch.cat((abstract_f,h_abstract_f),dim=0)
        kph_f = torch.cat((kph_f,h_kph_f),dim=0) 
        break
                          
    len_abstract = abstract_f.size(1)             
    all_rewards = D_model.calculate_rewards(abstract_f,kph_f,len_abstract,len_list_f,pred_str_new2dlist,total_len)            
    all_rewards = all_rewards.reshape(1,-1)
    print(all_rewards.size())    
        
            
            


def main():
    word2idx, idx2word, vocab = load_vocab(opt)
#    with suppress_stdout():
    tokenized_src = read_tokenized_src_file(src_file, remove_eos=False, title_guided=False)
    trg_str_2dlist = read_tokenized_trg_file(trg_file)
    tokenized_title = None
    print("test data loaded")   
    test_one2many_src = build_interactive_predict_dataset(tokenized_src, word2idx, idx2word, opt, tokenized_title)

    test_one2many_dataset = KeyphraseDataset(test_one2many_src, word2idx=word2idx, idx2word=idx2word,
                                    type='one2many', delimiter_type=opt.delimiter_type, load_train=False, remove_src_eos=opt.remove_src_eos, title_guided=opt.title_guided)
    print("########################################################################")    
    pred_str_2dlist = []
    src_list = []
    max_len = 0
    for abstract in test_one2many_dataset:
        x = torch.Tensor(abstract["src"]).to(devices).long()
        src_list.append(x)
    src_list = pad_sequence(src_list).t()
    for src_str_list in trg_str_2dlist:
        one_list = []
        for i,word_list in enumerate(src_str_list):
            one_list.append(convert_to_string_list(word_list,opt.word2idx))
        pred_str_2dlist.append(one_list) 
    hierarchal_attention_Discriminator(src_list,pred_str_2dlist)
            

            
        
            
            
 
main()           
            
    