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
#class Discriminator(nn.Module):
"""
src = [2,4,5,6,3,1,4,2]
kph = [[2,4,2,4],[2,2,1],[3,1],[2,3,5,6]]


def padded_all(target,total_kphs,pad_id):
    #target =  [[1,2,3], [2,4,5,6], [2,4,6,7,8]]
    max_cols = max([len(row) for row in target])
    max_rows = total_kphs
    padded = [batch + [[pad_id] * (max_cols)] * (max_rows - len(batch)) for batch in target]
    padded = torch.tensor([row + [pad_id] * (max_cols - len(row)) for row in padded])
    padded = padded.view(-1, max_rows, max_cols)
    padded = padded[0]
    return padded
    
def transform_src(src,total_kphs):
    src = torch.Tensor(src * total_kphs)
    src = src.reshape(total_kphs,-1)
    return src
    
def train_one_abstract(src,kph,reward_type,batch_size,loss_criterion):
    #phrase_reward = np.zeros((batch_size, max_num_phrases))
    #phrase_reward = 
    total_kphs = len(kph)
    src = transform_src(src)
    total_kphs = len(kph)
    kph = padded_all(kph,total_kphs,self.pad_idx)
    output = self.forward(src,kph)
    loss = loss_criterion(output,reward_type)
    return loss
        
        #output = self.forward(src,kph)
        #loss = Loss(output,reward_type)
        #total_loss+=loss

    
def train_one_batchs(src_str_list,pred_str_2dlist, trg_str_2dlist, batch_size):
    total_abstract_loss = 0
    
    for idx, (src_list, pred_str_list,target_str_list) in enumerate(zip(src_str_list, pred_str_2dlist,trg_str_2dlist)):
        abstract_loss_real = train_one_abstract(src_list,pred_str_list,1,batch_size)
        abstract_loss_fake = train_one_abstract(src_list,pred_str_list,0,batch_size)
        total_abstract_loss+= abstract_loss_real + abstarct_loss_fake
    return (total_abstract_loss/batch_size)


def train_one_batch(one2many_batch, generator, optimizer):
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, title, title_oov, title_lens, title_mask = one2many_batch
    src = src.to(self.device)
    src_mask = src_mask.to(self.device)
    src_oov = src_oov.to(self.device)   
    eos_idx = self.eos_idx
    sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask, entropy, location_of_eos_for_each_batch, location_of_peos_for_each_batch = generator.sample(
        src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=False, one2many=one2many,
        one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=perturb_std, entropy_regularize=entropy_regularize, title=title, title_lens=title_lens, title_mask=title_mask)
    pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx, delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                              src_str_list, opt.separate_present_absent, pykp.io.PEOS_WORD)
    total_abstract_loss = 0
    for idx, (src_list, pred_str_list,target_str_list) in enumerate(zip(src_str_list, pred_str_2dlist,trg_str_2dlist)):
        abstract_loss_real = train_one_abstract(src_list,pred_str_list,1,batch_size)
        abstract_loss_fake = train_one_abstract(src_list,pred_str_list,0,batch_size)
        total_abstract_loss+= abstract_loss_real + abstarct_loss_fake
    avg_batch_loss = (total_abstract_loss/batch_size)
   
    avg_batch_loss.backward()
    optimizer.step()
   
    return avg_batch_loss.detach().item()

""" 

#The Summary is tensor([14962,  2108,    68,     6,  9602,    10,  7330, 10033,    22,    11,
#           59, 35884,    28,  1059,   584,   722,   128,   179,    36,     7,
#           13,    84,    12,  1119,     6,   299,  2108,   409,    68,     6,
#         9602,    10,  7330, 10033,    13,     6,  1242,  5669,    37,    10,
#           12,  1527,    11,    81,   912,    85,   409,    15,  1437,     9,
#           11,    59, 35884,    28,     3,    17,  1059,   584,   722,   128,
#          179,    16,    85,   409,  1324,     6, 20681,  7330, 10033,    10,
#         7182,  6638,    10,     6, 14962,   110,    14,    43,     7,     6,
#         9602, 25676,    13,     6,  5144,   477,     9,   171,     6,  7330,
#        10033,    24,   282,    12, 10359,   343,     7,     6, 14962,   110,
#           14,   342,    27,     6, 35884,   409,     7, 35884,   236, 16287,
#            6,  1249,  9283,    14,   463,    12,   788,     6,   601,     8,
#         7632,    12,     6,   570,     7,   710,   153,  1661,    74,    71,
#           56,    67,    43,    12,  2103,     6,    74,    10,    85,   409,
#           13,     6,  9602,    10,  7330, 10033,  3076,     9,    39,  2993,
#           18,     6,  2108,   409,    68,   335,    57,    63,  1739,   596,
#          845,     7,    13,    49,    43,    85,   409,     9,     6, 14962,
#          110,    10,     6,  2108,    68,     6,  9602,    10,  7330, 10033,
#           24,   342,     9,    10,    22,    11, 11506,   604,     9,    11,
#            3, 14428,  3287,   944,    13,     6,  7330, 10033,    83,  1982,
#           13,     6, 10359,    55,     7,    19,    93,     6,  1128,    54,
#          238,    10,  2103,   145,   263,    85,    10,    96,    32,    29,
#          627,    22,    20,   186, 35884,    28,     3,   238,     7,   576,
#            9,    19,   163,    82,    51,    21,     6,  1621,    85,   292,
#            8,    11,    48, 10526,   305,  1627,   672,    17,     3,    16,
#            9,    35,  6555,     6,  1305,  1661,    74,    10,    85,     8,
#            6,   232,   976,     7,    13,    84,    12,   788,     6,   325,
#            8,     6, 14962,   110,     9,    98,   428,  5241,   270,     6,
#         1621,    85,    45,    56,    67,  1310,     7,     2,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0], device='cuda:1')
#All the required keyphrases are  [[9602], [7330, 10033], [1059, 584, 722, 128, 179], [1242, 5669, 37], [48, 10526, 305, 1627, 672], [14025]]
#The Predicted Keyphrases are [[35884, 28, 1059, 584, 722, 128, 179], [14962, 110], [1361], [1621], [28, 2159, 85], [9602, 25676, 30], [123, 35884, 8, 9602, 25676]]
#

#src = [14962,  2108,    68,     6,  9602,    10,  7330, 10033,    22,    11,
#           59, 35884,    28,  1059,   584,   722,   128,   179,    36,     7,
#           13,    84,    12,  1119,     6,   299,  2108,   409,    68,     6,
#         9602,    10,  7330, 10033,    13,     6,  1242,  5669,    37,    10,
#           12,  1527,    11,    81,   912,    85,   409,    15,  1437,     9,
#           11,    59, 35884,    28,     3,    17,  1059,   584,   722,   128,
#          179,    16,    85,   409,  1324,     6, 20681,  7330, 10033,    10,
#         7182,  6638,    10,     6, 14962,   110,    14,    43,     7,     6,
#         9602, 25676,    13,     6,  5144,   477,     9,   171,     6,  7330,
#        10033,    24,   282,    12, 10359,   343,     7,     6, 14962,   110,
#           14,   342,    27,     6, 35884,   409,     7, 35884,   236, 16287,
#            6,  1249,  9283,    14,   463,    12,   788,     6,   601,     8,
#         7632,    12,     6,   570,     7,   710,   153,  1661,    74,    71,
#           56,    67,    43,    12,  2103,     6,    74,    10,    85,   409,
#           13,     6,  9602,    10,  7330, 10033,  3076,     9,    39,  2993,
#           18,     6,  2108,   409,    68,   335,    57,    63,  1739,   596,
#          845,     7,    13,    49,    43,    85,   409,     9,     6, 14962,
#          110,    10,     6,  2108,    68,     6,  9602,    10,  7330, 10033,
#           24,   342,     9,    10,    22,    11, 11506,   604,     9,    11,
#            3, 14428,  3287,   944,    13,     6,  7330, 10033,    83,  1982,
#           13,     6, 10359,    55,     7,    19,    93,     6,  1128,    54,
#          238,    10,  2103,   145,   263,    85,    10,    96,    32,    29,
#          627,    22,    20,   186, 35884,    28,     3,   238,     7,   576,
#            9,    19,   163,    82,    51,    21,     6,  1621,    85,   292,
#            8,    11,    48, 10526,   305,  1627,   672,    17,     3,    16,
#            9,    35,  6555,     6,  1305,  1661,    74,    10,    85,     8,
#            6,   232,   976,     7,    13,    84,    12,   788,     6,   325,
#            8,     6, 14962,   110,     9,    98,   428,  5241,   270,     6,
#         1621,    85,    45,    56,    67,  1310,     7,     2,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#            0,     0,     0,     0,     0,     0,     0]
#
#kph = [[9602], [7330, 10033], [1059, 584, 722, 128, 179], [1242, 5669, 37], [48, 10526, 305, 1627, 672], [14025]]

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

        
devices = "cuda:3"

        
        
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
        
        self.RNN1 = S_RNN(embedding_dim,hidden_dim,n_layers,True)  ### Abstract RNN
        self.RNN2 = S_RNN(embedding_dim,hidden_dim,n_layers,True)  ### Summary RNN
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.Compress = nn.Linear(2*hidden_dim,hidden_dim)
        self.attention = nn.Linear(hidden_dim,hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.MegaRNN = nn.GRU(hidden_dim,2*hidden_dim,n_layers)
        self.Linear = nn.Linear(2*hidden_dim,1)
        
        
        
        
    def padded_all(self,target,total_kphs,pad_id):
        #target =  [[1,2,3], [2,4,5,6], [2,4,6,7,8]]
        max_cols = max([len(row) for row in target])
        max_rows = total_kphs
        padded = [batch + [pad_id] * (max_rows - len(batch)) for batch in target]
        padded = torch.tensor([row + [pad_id] * (max_cols - len(row)) for row in padded])
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
    
    def calc_loss(self,output,target_type):
        total_len = output.size()
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
    
    def calc_rewards(self,output):      ## for maximizing f1 score 
        total_len = output.size()
        criterion = nn.BCEWithLogitsLoss()
        outputs = self.sigmoid(output)
        return outputs
    
    def calculate_context_rewards(self,abstract_t,kph_t,target_type,len_list):
        total_rewards = torch.Tensor([]).to(devices)
        total_rewards = total_rewards.unsqueeze(0)
        x = self.attention(abstract_t)
        temp = kph_t.permute(0,2,1)
        x = torch.bmm(x,temp)
        x = x.permute(0,2,1)
        x = torch.bmm(x,abstract_t)
        output = torch.cat((x,kph_t),dim=2)
        output = self.Linear(output)

        output = output.squeeze(2)
        total_reward,total_loss = 0,0
        for i,len_i in enumerate(len_list):
            r_output = output[i,:len_i].squeeze(0)
            reward = self.calc_rewards(r_output)
            reward = reward.unsqueeze(0)
            if(len(reward.size())==1):
                reward = reward.unsqueeze(0)
            total_rewards = torch.cat((total_rewards,reward),dim=1)
        total_rewards = total_rewards.squeeze(0)
        return total_rewards
        
    
    def calculate_context(self,abstract_t,kph_t,target_type,len_list):  ## for  maximizing f1 score
        x = self.attention(abstract_t)
        temp = kph_t.permute(0,2,1)
        x = torch.bmm(x,temp)
        x = x.permute(0,2,1)
        x = torch.bmm(x,abstract_t)
        output = torch.cat((x,kph_t),dim=2)
        output = self.Compress(output)
        output = output.squeeze(2)
        concat_output = torch.cat((abstract_t,kph_t),dim=1) 
        concat_output = concat_output.permute(1,0,2)
        x,hidden = self.MegaRNN(concat_output)
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



    