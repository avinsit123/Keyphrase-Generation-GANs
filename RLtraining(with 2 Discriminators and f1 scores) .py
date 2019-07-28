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
from Discriminator import Discriminator as D1
from Discriminator_individual import Discriminator as D2
import torch.nn.functional as F

def Doit(a,idx2word):
    s = " "
    for word in a:
        if idx2word[word]=="<eos>":
            break;
        s += idx2word[word] + " "
        
    return s


def DIT(b,idx2word):
    ar = []
    for sentence in b:
        s = " "
        #print(len(sentence))
        for word in sentence:
            s += idx2word[word]+ " "
        ar.append(s)
    print(len(sentence))
    return ar
        
    
#####################################################################################################
opt = argparse.Namespace(attn_mode='concat', baseline='self', batch_size=32, batch_workers=4, bidirectional=True, bridge='copy', checkpoint_interval=4000, copy_attention=True, copy_input_feeding=False, coverage_attn=False, coverage_loss=False, custom_data_filename_suffix=False, custom_vocab_filename_suffix=False, data='data/kp20k_tg_sorted', data_filename_suffix='', dec_layers=1, decay_method='', decoder_size=300, decoder_type='rnn', delimiter_type=0, delimiter_word='<sep>', device=device(type='cuda', index=2), disable_early_stop_rl=False, dropout=0.1, dynamic_dict=True, early_stop_tolerance=4, enc_layers=1, encoder_size=150, encoder_type='rnn', epochs=20, exp='kp20k.rl.one2many.cat.copy.bi-directional', exp_path='exp/kp20k.rl.one2many.cat.copy.bi-directional.20190701-192604', final_perturb_std=0, fix_word_vecs_dec=False, fix_word_vecs_enc=False, goal_vector_mode=0, goal_vector_size=16, gpuid=2, init_perturb_std=0, input_feeding=False, lambda_coverage=1, lambda_orthogonal=0.03, lambda_target_encoder=0.03, learning_rate=0.001, learning_rate_decay=0.5, learning_rate_decay_rl=False, learning_rate_rl=5e-05, loss_normalization='tokens', manager_mode=1, match_type='exact', max_grad_norm=1, max_length=60, max_sample_length=6, max_unk_words=1000, mc_rollouts=False, model_path='model/kp20k.rl.one2many.cat.copy.bi-directional.20190701-192604', must_teacher_forcing=False, num_predictions=1, num_rollouts=3, one2many=True, one2many_mode=1, optim='adam', orthogonal_loss=False, param_init=0.1, perturb_baseline=False, perturb_decay_factor=0.0001, perturb_decay_mode=1, pre_word_vecs_dec=None, pre_word_vecs_enc=None, pretrained_model='model/kp20k.ml.one2many.cat.copy.bi-directional.20190628-114655/kp20k.ml.one2many.cat.copy.bi-directional.epoch=2.batch=54573.total_batch=116000.model', regularization_factor=0.0, regularization_type=0, remove_src_eos=False, replace_unk=True, report_every=10, review_attn=False, reward_shaping=False, reward_type=7, save_model='model', scheduled_sampling=False, scheduled_sampling_batches=10000, seed=9527, separate_present_absent=True, share_embeddings=True, source_representation_queue_size=128, source_representation_sample_size=32, start_checkpoint_at=2, start_decay_at=8, start_epoch=1, target_encoder_size=64, teacher_forcing_ratio=0, timemark='20190701-192604', title_guided=False, topk='G', train_from='', train_ml=False, train_rl=True, truncated_decoder=0, use_target_encoder=False, vocab='data/kp20k_separated/', vocab_filename_suffix='', vocab_size=100002, warmup_steps=4000, word_vec_size=100, words_min_frequency=0)

### kp20k_tg_seperated

devices = "cuda:2"
alpha = 0.95
beta = 0.05
##### TUNE HYPERPARAMETERS ##############
hidden_dim = 150
embedding_dim = 200
n_layers = 2 
##  batch_reward_stat, log_selected_token_dist = train_one_batch(batch, generator, optimizer_rl, opt, perturb_std)
#########################################################
def Catter_D2(model,kph,rewards,total_len):
     lengths = [len(kp)+1 for kp in kph]
     max_len = max(lengths)
     x = torch.Tensor([])
     rewards_shape = rewards.repeat(max_len).reshape(-1,rewards.size(0)).t()
     x= torch.Tensor([])
     x = x.to(devices)
     for i,keyphrase in enumerate(rewards_shape):
         x = torch.cat((x,keyphrase[:lengths[i]]))
     x = F.pad(input=x,pad=(0,total_len-x.size(0)),mode='constant',value=0)
     return x 
    
def generate_rewards_D2(model,src,kph,total_len):
    total_kphs = len(kph)
    _,rewards = model.forward(kph,1,devices)
    rewards = model.sigmoid(rewards)
#    print("D2 ab:",rewards)
#    print("fsg3:",kph)
    reshaped_rewards = Catter_D2(model,kph,rewards,total_len)
#    print("D2:",reshaped_rewards)
    return reshaped_rewards

def Catter_D1(model,kph,rewards,total_len):
     lengths = [len(kp)+1 for kp in kph]
     max_len = max(lengths)
     x = torch.Tensor([])
     rewards_shape = rewards.repeat(max_len).reshape(-1,rewards.size(0)).t()
     x= torch.Tensor([])
     x = x.to(devices)
     for i,keyphrase in enumerate(rewards_shape):
         x = torch.cat((x,keyphrase[:lengths[i]]))
     x = F.pad(input=x,pad=(0,total_len-x.size(0)),mode='constant',value=0)
     return x 
    
def generate_rewards_D1(model,src,kph,total_len):
    total_kphs = len(kph)
    temp_kph = kph
    src = model.transform_src(src,total_kphs)
    kph = model.padded_all(kph,total_kphs,model.pad_idx)
    src,kph = src.long().to(devices),kph.long().to(devices)
    rewards = model.forward(src,kph)
    rewards = model.sigmoid(rewards)
#    print("D1 ab:",rewards)
#    print("fsg1:",kph)
    reshaped_rewards = Catter_D2(model,temp_kph,rewards,total_len)
#    print("D1:",reshaped_rewards)
    return reshaped_rewards

def convert_to_string_list(kph,idx2word):
#    print("The jph is",kph)
    df = []
    
    
    for sent in kph:
        ar = []
        for kps in sent:
            s = []
            for insd in kps:
#                print(insd)
                s.append(idx2word[insd])
            ar.append(s)
        df.append(ar)
#    print("The df is",df)
#    sys.exit()
    return df
def train_one_batch(D_model_1,D_model_2,one2many_batch, generator, opt,perturb_std):
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
    reward_shaping = True
    baseline = opt.baseline
    match_type = opt.match_type
    regularization_type = opt.regularization_type ## DNT
    regularization_factor = opt.regularization_factor ##DNT
    
    if regularization_type == 2:
        entropy_regularize = True
    else:
        entropy_regularize = False
        
    if entropy_regularize:
        entropy_array = entropy.data.cpu().numpy()
    else:
        entropy_array = None
    
    start_time = time.time()
    sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask, entropy, location_of_eos_for_each_batch, location_of_peos_for_each_batch = generator.sample(
        src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=False, one2many=one2many,
        one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=perturb_std, entropy_regularize=entropy_regularize, title=title, title_lens=title_lens, title_mask=title_mask)
    
    #print(log_selected_token_dist.size())
    pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx, delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                              src_str_list, opt.separate_present_absent, pykp.io.PEOS_WORD)
    temp = pred_str_2dlist
    pred_str_2dlist = convert_to_string_list(pred_str_2dlist,opt.idx2word)
    
    
    
    total_len = log_selected_token_dist.size(1)
    max_num_pred_phrases = max([len(pred_str_list) for pred_str_list in pred_str_2dlist])
    
    reward_shaping = True
    aphrase_reward = compute_phrase_reward(pred_str_2dlist, trg_str_2dlist, batch_size, max_num_pred_phrases, reward_shaping,
                              reward_type, topk, match_type, regularization_factor, regularization_type, entropy_array)
    phrase_reward = shape_reward(aphrase_reward)
    stepwise_reward = phrase_reward_to_stepwise_reward(phrase_reward, pred_eos_idx_mask)
   # print(len(stepwise_reward[0]),len(stepwise_reward),log_selected_token_dist.size())
    stepwise_reward = torch.Tensor(stepwise_reward)
    pred_str_2dlist = temp
    target_str_2dlist = convert_list_to_kphs(trg)
  #  log_selected_token_dist =  log_selected_token_dist.to("cuda:")
    """
     src = [batch_size,abstract_seq_len]
     target_str_2dlist = list of list of true keyphrases
     pred_str_2dlist = list of list of false keyphrases
    
    """
    
    total_abstract_loss = 0
    batch_mine = 0
    all_rewards = 0
    for idx, (src_list, pred_str_list,target_str_list) in enumerate(zip(src, pred_str_2dlist,target_str_2dlist)):

#        print("dsafds",Doit(list(src_list.cpu().numpy()),opt.idx2word))
#        print("fdsg",DIT(pred_str_list,opt.idx2word))
#        print("sadsf",DIT(target_str_list,opt.idx2word))
#        continue
        batch_mine+=1
        if (len(target_str_list)==0 or len(pred_str_list)==0):
            continue
        #print("ewre",src.size(),len(target_str_list),target_str_list)
#        reshaped_rewards_1 = generate_rewards_D1(D_model_1,src_list,pred_str_list,total_len)
#        reshaped_rewards_2 = generate_rewards_D2(D_model_2,src_list,pred_str_list,total_len)
        reshaped_rewards_1 = generate_rewards_D1(D_model_1,src_list,pred_str_list,total_len)
        reshaped_rewards_2 = generate_rewards_D2(D_model_2,src_list,pred_str_list,total_len)
        reshaped_rewards = alpha * reshaped_rewards_1 + beta * reshaped_rewards_2
       # print(reshaped_rewards_1.size()==reshaped_rewards_2.size())
       # print(reshaped_rewards)
        if idx==0:
            all_rewards = reshaped_rewards
            all_rewards = all_rewards.unsqueeze(0)
        else:
            #print(all_rewards.size(),reshaped_rewards.size())
            reshaped_rewards = reshaped_rewards.unsqueeze(0)
            #print(all_rewards.size(),reshaped_rewards.size())
            all_rewards = torch.cat((all_rewards,reshaped_rewards),dim=0)
    #print(all_rewards)
    
    log_selected_token_dist =  log_selected_token_dist.to(devices)
    log_selected_token_dist = log_selected_token_dist[:all_rewards.size(0),:]
    stepwise_reward = stepwise_reward[:all_rewards.size(0),:]
    all_rewards = (all_rewards + stepwise_reward.float().to(devices))/2
    calculated_rewards = log_selected_token_dist * all_rewards
    individual_rewards = torch.sum(calculated_rewards,dim=1)
#    print(log_selected_token_dist[0],all_rewards[0])
#    print(calculated_rewards[0])
#    print(individual_rewards[0])
    J = torch.mean(individual_rewards)
    return J
    

def main():
    #print("agsnf efnghrrqthg")
    print("dfsgf")
    clip = 5
    start_time = time.time()
    train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_data_and_vocab(opt, load_train=True)
    load_data_time = time_since(start_time)
    logging.info('Time for loading the data: %.1f' % load_data_time)
    
    model = Seq2SeqModel(opt)
    #model = model.device()
    #print("The Device is",opt.gpuid)
    
    #model.load_state_dict(torch.load("model/kp20k.ml.one2many.cat.copy.bi-directional.20190704-170553/kp20k.ml.one2many.cat.copy.bi-directional.epoch=2.batch=264.total_batch=8000.model"))
   # model.load_state_dict(torch.load("Checkpoint_individual_3.pth.tar"))
    model.load_state_dict(torch.load("model/kp20k.ml.one2many.cat.copy.bi-directional.20190715-132016/kp20k.ml.one2many.cat.copy.bi-directional.epoch=3.batch=26098.total_batch=108000.model",map_location="cuda:2"))  
    
    if torch.cuda.is_available():
        model = model.to(devices)
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
    
    D_model_1 = D1(opt.vocab_size,embedding_dim,hidden_dim,n_layers,opt.word2idx[pykp.io.PAD_WORD])
    D_model_2 = D2(opt.vocab_size,embedding_dim,hidden_dim,n_layers,opt.word2idx[pykp.io.PAD_WORD])    
  
    # D_model.load_state_dict(torch.load("Discriminator_checkpts/Checkpoint_Individual_Training_4.pth.tar"))

    D_model_1.load_state_dict(torch.load("Discriminator_checkpts/D_model_combined1.pth.tar"))
    D_model_2.load_state_dict(torch.load("Discriminator_checkpts/Checkpoint_Individual_Training(tg_guided)_5.pth.tar")) 
    
    PG_optimizer = torch.optim.Adagrad(model.parameters(),0.00005)
    
    print("The Discriminator 1 statistics are ",D_model_1)
    print("The Discriminator 2 statistics are ",D_model_2)
    
#    sys.exit()
    if torch.cuda.is_available():
        D_model_1 = D_model_1.to(devices)
        D_model_2 = D_model_2.to(devices)   
        
    

    total_epochs = 5
    for epoch in range(total_epochs):
        
        total_batch = 0
        print("Starting with epoch:",epoch)
        for batch_i, batch in enumerate(train_data_loader):
            total_batch+=1

            PG_optimizer.zero_grad() 
            
            if perturb_decay_mode == 0:  # do not decay
                perturb_std = init_perturb_std
            elif perturb_decay_mode == 1:  # exponential decay
                perturb_std = final_perturb_std + (init_perturb_std - final_perturb_std) * math.exp(-1. * total_batch * perturb_decay_factor)
            elif perturb_decay_mode == 2:  # steps decay
                perturb_std = init_perturb_std * math.pow(perturb_decay_factor, math.floor((1+total_batch)/4000))


            avg_rewards = train_one_batch(D_model_1,D_model_2,batch,generator,opt,perturb_std)
           
            avg_rewards.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            
            PG_optimizer.step()

            if batch_i%100==0:
                print("Saving the file ...............----------->>>>>")        
                print("The avg reward is",-avg_rewards.item())
                state_dfs = model.state_dict()
                torch.save(state_dfs,"RL_Checkpoints/Checkpoint(2_Disc_combined)(f1_scores)" + str(epoch) + ".pth.tar")
     

            
        
        print("Saving the file ...............----------->>>>>")        
        state_dfs = model.state_dict()
        torch.save(state_dfs,"RL_Checkpoints/Checkpoint(2_Disc_combined)(f1_scores)" + str(epoch) + ".pth.tar")

######################################            
            
        
main()


    

""""
kph = [[785, 472, 567], [109, 41, 3720], [2209], [379, 441]]
lengths = [len(kp)+1 for kp in kph]
 max_len = max(lengths)
 # rewards = torch.randn(4)
 rewards_shape = rewards.repeat(max_len).reshape(-1,max_len).t()
 
#lengths = [4, 4, 2, 3]
x = torch.Tensor([])
for i,keyphrase in enumerate(rewards_reshape):
    x = torch.cat(x,keyphrase[:lengths[i]])
return x 

""" 
    
    
