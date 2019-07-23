#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:38:18 2019

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
import configx
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
from Discriminator_Softmax import Discriminator

#####################################################################################################
opt = argparse.Namespace(attn_mode='concat', baseline='self', batch_size=32, batch_workers=4, bidirectional=True, bridge='copy', checkpoint_interval=4000, copy_attention=True, copy_input_feeding=False, coverage_attn=False, coverage_loss=False, custom_data_filename_suffix=False, custom_vocab_filename_suffix=False, data='data/kp20k_separated/', data_filename_suffix='', dec_layers=1, decay_method='', decoder_size=300, decoder_type='rnn', delimiter_type=0, delimiter_word='<sep>', device=device(type='cuda', index=2), disable_early_stop_rl=False, dropout=0.1, dynamic_dict=True, early_stop_tolerance=4, enc_layers=1, encoder_size=150, encoder_type='rnn', epochs=20, exp='kp20k.rl.one2many.cat.copy.bi-directional', exp_path='exp/kp20k.rl.one2many.cat.copy.bi-directional.20190701-192604', final_perturb_std=0, fix_word_vecs_dec=False, fix_word_vecs_enc=False, goal_vector_mode=0, goal_vector_size=16, gpuid=1, init_perturb_std=0, input_feeding=False, lambda_coverage=1, lambda_orthogonal=0.03, lambda_target_encoder=0.03, learning_rate=0.001, learning_rate_decay=0.5, learning_rate_decay_rl=False, learning_rate_rl=5e-05, loss_normalization='tokens', manager_mode=1, match_type='exact', max_grad_norm=1, max_length=60, max_sample_length=6, max_unk_words=1000, mc_rollouts=False, model_path='model/kp20k.rl.one2many.cat.copy.bi-directional.20190701-192604', must_teacher_forcing=False, num_predictions=1, num_rollouts=3, one2many=True, one2many_mode=1, optim='adam', orthogonal_loss=False, param_init=0.1, perturb_baseline=False, perturb_decay_factor=0.0001, perturb_decay_mode=1, pre_word_vecs_dec=None, pre_word_vecs_enc=None, pretrained_model='model/kp20k.ml.one2many.cat.copy.bi-directional.20190628-114655/kp20k.ml.one2many.cat.copy.bi-directional.epoch=2.batch=54573.total_batch=116000.model', regularization_factor=0.0, regularization_type=0, remove_src_eos=False, replace_unk=True, report_every=10, review_attn=False, reward_shaping=False, reward_type=7, save_model='model', scheduled_sampling=False, scheduled_sampling_batches=10000, seed=9527, separate_present_absent=True, share_embeddings=True, source_representation_queue_size=128, source_representation_sample_size=32, start_checkpoint_at=2, start_decay_at=8, start_epoch=1, target_encoder_size=64, teacher_forcing_ratio=0, timemark='20190701-192604', title_guided=False, topk='G', train_from='', train_ml=False, train_rl=True, truncated_decoder=0, use_target_encoder=False, vocab='data/kp20k_separated/', vocab_filename_suffix='', vocab_size=50002, warmup_steps=4000, word_vec_size=100, words_min_frequency=0)


hidden_dim = 150
embedding_dim = 200
n_layers = 2 
clip = 5 

def main():
      clip = 5
    start_time = time.time()
    train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_data_and_vocab(opt, load_train=True)
    load_data_time = time_since(start_time)
    