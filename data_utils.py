#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:44:59 2020

@author: raghuramkowdeed
"""

import pandas as pd
import numpy as np

import logging
import math
import os

import torch

from functools import partial 


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from tqdm import tqdm

def _read_data(f, n) :
    x = pd.read_csv(f, nrows = n, usecols=['id', 'comment_text', 'target'])
    a = x['target']
    a[a>=0.5] = 1
    a[a<0.5] = 0
    x['target'] = a
    
    return x



def get_device() :
    if torch.cuda.is_available():  
       dev = "cuda:0" 
    else:  
       dev = "cpu" 
    device = torch.device(dev)      

    return device

def assign_GPU(Tokenizer_output, device):
    tokens_tensor = Tokenizer_output['input_ids'].to(device)
    #token_type_ids = Tokenizer_output['token_type_ids'].to(device)
    attention_mask = Tokenizer_output['attention_mask'].to(device)


    output = {'input_ids' : tokens_tensor, 
          #'token_type_ids' : token_type_ids, 
          'attention_mask' : attention_mask}

    return output 
   

def get_tokenizer() :
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case= True)
    return tokenizer
    

def bal_collate(i, x_pos, x_neg, tokenizer, device):
    
    
    pos_sample = x_pos.iloc[i, :]
    neg_sample = x_neg.sample(n=pos_sample.shape[0],replace=False)
    
    data = pd.concat( [pos_sample, neg_sample], axis=0, ignore_index=True)
    

    txt = data['comment_text'].values.tolist()
    txt_id = data['id'].values.tolist()
    y = data['target'].values.tolist()
    
    
    batch = assign_GPU( tokenizer(txt, padding=True, truncation=True, max_length=256, pad_to_max_length = True, add_special_tokens = True,
                                  return_tensors="pt"), device )
    batch['target'] = torch.tensor(y, dtype=torch.long,device=device)
    
    sample_weights = torch.ones_like(batch['target'], device=device) 

    ind_0 = ( batch['target'] ==0 ).nonzero()
    ind_1 = ( batch['target'] !=0 ).nonzero()

    sample_weights[ind_0] = ind_1.shape[0]
    sample_weights[ind_1] = ind_0.shape[0]
    sample_weights = sample_weights *1.0/sample_weights.sum(axis=0, keepdims=True)  
 

    
    return batch['input_ids'], batch['attention_mask'], batch['target'], sample_weights, txt, txt_id


def get_data_loader_bal(x_pos_file, x_neg_file, batch_size=16, 
                        nrows_pos= None, nrows_neg = None, mode = 'train', tokenizer = None):
    x_pos = _read_data(x_pos_file, nrows_pos)
    x_pos = x_pos[x_pos['comment_text'].str.len() > 0 ]
    
    x_neg = _read_data(x_neg_file, nrows_neg)
    x_neg = x_neg[x_neg['comment_text'].str.len() > 0 ]
    
    data_ind= range(x_pos.shape[0])
    
    if mode == 'train' :        
        sampler = RandomSampler(data_ind)
        shuffle = True
    else :
        sampler = SequentialSampler(data_ind)
        shuffle = False

        
    #tokenizer = get_tokenizer()
    device = get_device()
    
    dataloader = DataLoader(  data_ind, sampler = sampler,
                                    collate_fn=partial(bal_collate, x_pos=x_pos, x_neg=x_neg, tokenizer=tokenizer, device=device), 
                                    batch_size=batch_size,  )
    
    return dataloader
    
    


def test_collate(i, x_test, tokenizer, device,nrows = None):
    
    
    
    data = x_test.iloc[i, :]

    txt = data['comment_text'].values.tolist()
    txt_id = data['id'].values.tolist()
    
    
    batch = assign_GPU( tokenizer(txt, padding=True, truncation=True, max_length=256, pad_to_max_length = True, add_special_tokens = True,
                                  return_tensors="pt"), device )
    
    
    

    
    return batch['input_ids'], batch['attention_mask'], txt, txt_id


def get_data_loader_pred(test_file, tokenizer, nrows=None):
    x_test = pd.read_csv(test_file, nrows = nrows)
    x_test = x_test[x_test['comment_text'].str.len() > 0 ]
    
    data_ind = range(x_test.shape[0])
    device = get_device()
    
    test_dataloader = DataLoader(  data_ind,
                                 collate_fn = partial(test_collate, x_test=x_test, tokenizer=tokenizer, 
                                                      device=device),
                                                      batch_size=16, shuffle=False,  )
    
    return test_dataloader

    
    
    
        
    
def get_data_pred(dataloader, model,  out_file, ) :
    
    
    pred_labels = []
    ids = []
    txts = []
    model.eval()
    for test_data in tqdm( dataloader ) :       

       with torch.no_grad() :
         
         prob = model.forward(input_ids=test_data[0],attention_mask=test_data[1], mode='eval')
         
         prob = prob.cpu().detach().numpy() 
         
         pred_labels.append(prob)
         ids.append(test_data[-1])
         txts.append(test_data[-2])
         
    ids = np.concatenate(ids)
    pred_labels = np.concatenate(pred_labels)
    txts = np.concatenate(txts)
    
    df = pd.DataFrame()
    df['id'] = ids
    df['txt'] = txts
    df['pred'] = pred_labels
    
    
    
    df.to_csv(out_file, index=False)
    
    return df


    
    
    
    
    
    
    
    
