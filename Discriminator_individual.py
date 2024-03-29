#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:36:18 2019

@author: r17935avinash
"""

from torch import nn as nn
import torch
import sys

class Discriminator(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,n_layers,pad_idx,bidirectional=False):
        super(Discriminator,self).__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.RNN = nn.GRU(embedding_dim,hidden_dim,n_layers)
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.Linear = nn.Linear(hidden_dim,1)
        self.sigmoid = nn.Sigmoid()
        
    def padded_all(self,target,total_kphs,pad_id,devices):
        max_cols = max([len(row) for row in target])
        max_rows = total_kphs
        #padded = [[pad_id] * (max_rows - len(batch)) for batch in target]
#        print("sDAS:",[[pad_id] * (max_rows - len(batch)) for batch in target])
        #print("adsgf:",padded)
        padded = torch.tensor([row + [pad_id] * (max_cols - len(row)) for row in target])
       # sys.exit()
        if torch.cuda.is_available():
            padded = padded.to(devices)
        return padded
    
    def forward(self,kph,target_type,devices="cuda:1"):
        total_kphs = len(kph)
        kph = self.padded_all(kph,total_kphs,self.pad_idx,devices)
        kph = kph.long()
        x = self.embedding(kph)
        x = x.permute(1,0,2)
        x,hidden = self.RNN(x)
        x = x.permute(1,0,2)
        hidden = hidden[1,:,:]
        output = self.Linear(hidden)
        output = output.squeeze(1)
        total_len = output.size(0)
        if target_type==1:
            results = torch.ones(total_len)*0.9
            if torch.cuda.is_available():
                results = results.to(devices)
        else:
            results = torch.zeros(total_len)
            if torch.cuda.is_available():
                results = results.to(devices)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output,results)
        #print("deebg is:",self.sigmoid(output))
        rewards = self.sigmoid(output)
        return loss,rewards
     
        