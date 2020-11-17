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

def get_idn_col() :
    idn_cols = [ 'asian', 'atheist', 'bisexual',
       'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',
       'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability',
       'jewish', 'latino', 'male', 'muslim', 'other_disability',
       'other_gender', 'other_race_or_ethnicity', 'other_religion',
       'other_sexual_orientation', 'physical_disability',
       'psychiatric_or_mental_illness', 'transgender', 'white']
    
    return idn_cols

def resample_idn(x, z0 = 0.5, z1 = 0.5):
    idn_col = get_idn_col()
    x = x.copy()
    idn_fea = x[idn_col].max(axis=1)
    x_idn = x[idn_fea.notnull()]
    x_no_idn = x[idn_fea.isnull()]
    
    idn_fea = idn_fea[idn_fea.notnull()]

    z = idn_fea.copy()
    z[idn_fea>=0.5] = 1
    z[idn_fea<0.5] = 0
    
    x_z0 = x_idn[z==0]
    x_z1= x_idn[z==1]

    x_z0 = x_z0.sample(n=int(x_idn.shape[0]*z0), replace=True)
    x_z1 = x_z1.sample(n=int(x_idn.shape[0]*z1), replace = True)
    
    x = pd.concat( [x_no_idn, x_z0, x_z1], axis=0)
    
    return x 


def _read_data(f, n, z0 = 0.5, z1 = 0.5) :
    
    
    x = pd.read_csv(f, nrows = n,)    
    
    x = resample_idn(x)    
    
    a = x['target']
    #a[a>=0.5] = 1
    #a[a<0.5] = 0
    x['target'] = a
    
    idn_col = get_idn_col()    
    b = x[idn_col].max(axis=1)
    #b[b>=0.5] = 1
    #b[b<0.5] = 0
    b = b.fillna(-1)
    x['idn_score'] = b
    
    
    
    x = x[['id', 'comment_text', 'target', 'idn_score']]
    
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
    neg_sample = x_neg.sample(n=pos_sample.shape[0],replace=True)
    
    data = pd.concat( [pos_sample, neg_sample], axis=0, ignore_index=True)
    

    txt = data['comment_text'].values.tolist()
    txt_id = data['id'].values.tolist()
    y = data['target'].values.tolist()
    z = data['idn_score'].values.tolist()
    
    
    batch = assign_GPU( tokenizer(txt, padding=True, truncation=True, max_length=256, pad_to_max_length = True, add_special_tokens = True,
                                  return_tensors="pt"), device )
    batch['target'] = torch.tensor(y, dtype=torch.float, device=device)
    batch['idn_score'] = torch.tensor(z, dtype=torch.float, device=device)
    
    sample_weights = torch.ones_like(batch['target'], device=device) 

    ind_0 = ( batch['target'] ==0 ).nonzero()
    ind_1 = ( batch['target'] !=0 ).nonzero()

    sample_weights[ind_0] = ind_1.shape[0]
    sample_weights[ind_1] = ind_0.shape[0]
    sample_weights = sample_weights *1.0/sample_weights.sum(axis=0, keepdims=True)  
 

    
    return batch['input_ids'], batch['attention_mask'], batch['target'], sample_weights, batch['idn_score'],txt, txt_id, 




def get_data_loader_bal(x_pos_file, x_neg_file, batch_size=16, 
                        nrows_pos= None, nrows_neg = None, mode = 'train', tokenizer = None):
    
    
    x_pos = _read_data(x_pos_file, nrows_pos, z0 = 0.5, z1 = 0.5)
    #_read_data(x_pos_file, nrows_pos)
    x_pos = x_pos[x_pos['comment_text'].str.len() > 0 ]
    
    #x_pos = resample_idn(x_pos)
    
    
    x_neg = _read_data(x_neg_file, nrows_neg, z0 = 0.5, z1 = 0.5)
    #_read_data(x_neg_file, nrows_neg)
    x_neg = x_neg[x_neg['comment_text'].str.len() > 0 ]
    
    #x_neg = resample_idn(x_neg)
    
    
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


    
    
    
    
    
    
    
    
