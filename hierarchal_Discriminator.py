#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:57:57 2019

@author: r17935avinash
"""


import torch 
from pykp.reward import *
from torch import nn as nn
import torch.nn.functional as F

def matmul(X, Y):
    taken = []
    for i in range(X.size(2)):
        result = (X[:,:,i]*Y)
        taken.append(result)
        results = torch.stack(taken,dim=2)
    return results

class S_RNN(nn.Module):
    def __init__(self,embed_dim,hidden_dim,n_layers,bidirectional=False):
        super(S_RNN,self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.RNN = nn.GRU(embed_dim,hidden_dim,n_layers)
        
        
    def forward(self,x):
        #x = self.embedding(x)
        #print(x.size())
        x = x.permute(1,0,2)
        x,hidden = self.RNN(x)
        x = x.permute(1,0,2)
        hidden = hidden.permute(1,0,2)
        return x,hidden
    
    
        



 #   fake_labels = torch.from_numpy(np.random.uniform(0, 0.3, size=(BATCH_SIZE))).float().to(DEVICE)
 #   real_labels = torch.from_numpy(np.random.uniform(0.7, 1.2, size=(BATCH_SIZE))).float().to(DEVICE)

        
devices = "cpu"

        
        
class Discriminator(S_RNN):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,n_layers,pad_idx,bidirectional=False):
        super().__init__(embedding_dim,hidden_dim,n_layers,bidirectional=False)
        #### INDICES INITIALIZATION
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        ##### MODEL SPECIFICATIONS
        self.RNN1 = S_RNN(embedding_dim,hidden_dim,n_layers,True)  ### Abstract RNN
        self.RNN2 = S_RNN(embedding_dim,hidden_dim,n_layers,True)  ### Summary RNN
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.MegaRNN = nn.GRU(hidden_dim,2*hidden_dim,n_layers)
        self.Linear = nn.Linear(2*hidden_dim,1)
        self.sigmoid = nn.Sigmoid()
        
        
        
    def padded_all(self,target,total_kphs,pad_id):
        #target =  [[1,2,3], [2,4,5,6], [2,4,6,7,8]]
        max_cols = max([len(row) for row in target])
        #print(max_cols)
        max_rows = total_kphs
        padded = [batch + [pad_id] * (max_rows - len(batch)) for batch in target]
        padded = torch.tensor([row + [pad_id] * (max_cols - len(row)) for row in padded])
        #print(padded)
        #padded = padded.view(-1, max_rows, max_cols)
        #padded 
        padded = padded.to(devices)
        return padded
        
    def transform_src(self,src,total_kphs):
        src = torch.Tensor(np.tile(src.cpu().numpy(),total_kphs))
        src = src.reshape(total_kphs,-1)
        src = src.to(devices)
        return src        
        
    def forward(self,src,kph):
        src = self.embedding(src)
        kph = self.embedding(kph)
        
        abstract_d = self.RNN1(src)[0]
        keyphrase_d = self.RNN2(kph)[1]
        keyphrase_d = keyphrase_d[:,0,:]
        abstract = abstract_d[0,:,:]
        return abstract,keyphrase_d
        

  
    def get_hidden_states(self,src,kph):
        total_kphs = len(kph)
        src = self.transform_src(src,total_kphs)
        kph = self.padded_all(kph,total_kphs,self.pad_idx)
        src,kph = src.long(),kph.long()
        h_abstract,h_kph = self.forward(src,kph)
        h_abstract,h_kph = h_abstract.unsqueeze(0),h_kph.unsqueeze(0) 
        return h_abstract,h_kph
        
    def calc_score_loss(self,inp,target_type):
        x = inp
        x = x.permute(1,0,2)
        x,hidden = self.MegaRNN(x)
        x = x[-1,:,:]
        output = self.Linear(x)
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
        avg_outputs = torch.mean(self.sigmoid(output))
        loss = criterion(output,results)
        return avg_outputs,loss
    
    def calc_keyphrase_score(self,inp,start_len,len_list):
        severed_inputs = torch.Tensor([]).to(devices)
        x = inp
        x = x.permute(1,0,2)
        x,hidden = self.MegaRNN(x)
        x = x.permute(1,0,2)
        output = self.sigmoid(self.Linear(x))
        for i in range(x.size(0)):
           severed_inputs = torch.cat((severed_inputs,output[i,start_len:start_len+len_list[i],0]))
        return severed_inputs
        

        
        

 ## torch.Size([5, 400, 150]) torch.Size([5, 150, 1])   

    

        
        
    
    
                
#src_RNN = SRNN()
# optimizer_rl = torch.optim.Adam(list(src_RNN.parameters())+list(tgt_RNN.parameters()),lr=0.01)    
        
""" TEST_ASD
 keywords = [[2,3,4],[3,2],[4,5,3,4,2]]
 summary = [1,2,3,4,5,6]
 total_kphs = len(kph)
 
 src_RNN = SRNN(50000,200,150,2)
 tgt_RNN = SRNN(5000,200,150,2)
 tgt_RNN(kph.long())[0].size()
 hidden_d = tgt_RNN(kph.long())[1].size()
 keyphrase_d = hidden_d[:,0,:]
 keyphrase_d = keyphrase_d.squeeze(2)
 abstract_d = abstract_d.permute(1,0,2)
 cosine_results = torch.bmm(abstract_d,keyphrase_d).squeeze(2)  
 cosine_avgs = torch.mean(cosine_results,dim=1)
 
 cosine_results = torch.bmm(abstract_d,keyphrase_d).squeeze(2)  ## Size(n_kphs,max_len)
 cosine_avgs = torch.mean(cosine_results,dim=1)
 lossd = calculate_loss(cosine_avdgs,1) + calculate_loss(cosine_avdgs,0)
 
"""
        
D_model = Discriminator(50002,200,150,2,0) 



    