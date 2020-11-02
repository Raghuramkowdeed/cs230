#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:14:08 2020

@author: raghuramkowdeed
"""

import argparse

from bert_toxic import BertForToxic
from data_utils import get_data_loader_bal, get_device

import numpy as np
import torch
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from transformers import get_linear_schedule_with_warmup

from transformers import BertModel, BertConfig, AdamW
import os 

from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer


def _read_data(f, n) :
    x = pd.read_csv(f, nrows = n)
    a = x['target']
    a[a>=0.5] = 1
    a[a<0.5] = 0
    x['target'] = a
    x = x.sample(frac=1)
    
    return x


def assign_GPU(Tokenizer_output, device):
    tokens_tensor = Tokenizer_output['input_ids'].to(device)
    token_type_ids = Tokenizer_output['token_type_ids'].to(device)
    attention_mask = Tokenizer_output['attention_mask'].to(device)


    output = {'input_ids' : tokens_tensor, 
          'token_type_ids' : token_type_ids, 
          'attention_mask' : attention_mask}

    return output 

def my_collate(data):

    
    
    txt, y = zip(*data)
    batch = assign_GPU( tokenizer(txt, padding=True, truncation=True, return_tensors="pt"), device )
    batch['target'] = torch.tensor(y, dtype=torch.long, device=device)
    
    sample_weights = torch.ones_like(batch['target'], device=device) 

    ind_0 = ( batch['target'] ==0 ).nonzero()
    ind_1 = ( batch['target'] !=0 ).nonzero()

    sample_weights[ind_0] = ind_1.shape[0]
    sample_weights[ind_1] = ind_0.shape[0]
    sample_weights = sample_weights *1.0/sample_weights.sum(axis=0, keepdims=True)  

    if ind_0.shape[0] == 0 or ind_1.shape[0] == 0 :
      sample_weights = torch.ones_like(batch['target'], device=device) 
      sample_weights = sample_weights *1.0/sample_weights.sum(axis=0, keepdims=True)  

 
    
    
    return batch['input_ids'], batch['attention_mask'], batch['target'], sample_weights

def train_epoch( model, train_dataloader, dev_dataloader, optimizer, scheduler ):
    
    ## For each batch, we must reset the gradients
    ## stored by the model.   
    train_loss = 0
    count = 0    
    model.train()
    
    for train_data in tqdm( train_dataloader ):
        
        
        
        # clear gradients
        optimizer.zero_grad()
        # evoke model in training mode on batch
        
        loss, logits = model.forward(input_ids=train_data[0],attention_mask=train_data[1],  labels = train_data[2],
                                   )
        
        # compute loss w.r.t batch
        
        # pass gradients back, startiing on loss value
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        scheduler.step()
        
        train_loss += loss.item()
        count = count + 1

    train_loss = train_loss/count

    pred_labels = []
    target_labels = []
    probs = []
    dev_loss = 0.0
    thres = 0.5
    count = 0
    model.eval()
    for dev_data in tqdm( dev_dataloader ) :
       

       with torch.no_grad() :
         
         loss, logits = model.forward(input_ids=dev_data[0],attention_mask=dev_data[1],  labels = dev_data[2],
                                   )
         
         
         dev_loss += loss.item()
         count = count + 1
         
         logits = logits.detach().cpu().numpy()
         prob = np.argmax(logits, axis=1).flatten()

         pred_labels.append(prob)
         target = dev_data[2].cpu().detach().numpy() 
         
         target_labels.append(target)

    dev_loss = dev_loss / count 
    
    target_labels = np.concatenate(target_labels)
    pred_labels = np.concatenate(pred_labels)
    
    
    accuracy = accuracy_score(target_labels, pred_labels, )
    f1 = f1_score(target_labels, pred_labels, )
    
    precision = precision_score(target_labels, pred_labels, )
    recall = recall_score(target_labels, pred_labels, )
    
    stats = {'accuracy': accuracy, 'precision':precision, 'recall':recall, 'f1':f1, 'dev_loss':dev_loss, 'train_loss':train_loss }
    
    

    
    # return the total to keep track of how you did this time around
    return stats



def run_model(pos_train_file , neg_train_file, pos_dev_file , neg_dev_file , nrows_train, nrows_dev, epochs , out_dir) :
    batch_size = 16
    
    #x_train = _read_data('../data/train_bal.csv', nrows_train)
    #x_dev = _read_data('../data/dev_bal.csv', nrows_dev)
    
    #train_data = list( zip( x_train['comment_text'].values, x_train['target'].values  ))
    
    #train_dataloader = DataLoader(  train_data, 
    #                            collate_fn=my_collate, 
    #                            batch_size=batch_size , shuffle=True,  )
   # #

    #dev_data = list( zip( x_dev['comment_text'].values, x_dev['target'].values  ))
    
    #dev_dataloader = DataLoader(  dev_data, 
    #                            collate_fn=my_collate, 
    #                            batch_size=batch_size, shuffle=False,  )
    
    
    
    train_dataloader = get_data_loader_bal(pos_train_file, neg_train_file, batch_size=batch_size, nrows_pos= nrows_train, nrows_neg = nrows_train, mode = 'train')
    dev_dataloader =   get_data_loader_bal(pos_dev_file, neg_dev_file, batch_size=batch_size, nrows_pos= nrows_dev, nrows_neg = nrows_dev, mode = 'dev')
    
    
    device = get_device()
    
    bert_hidden_states = 4
    config = BertConfig()
    config.output_hidden_states = True
    
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    model = model.to(device)
    
    
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
    
    
    total_steps = len(train_dataloader) * epochs
    
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    
    
    if not os.path.exists(out_dir) :
          os.makedirs(out_dir)
          
    best_score = -np.inf      
    
    stats_vec = []
    for epoch in range(epochs):
        stats = train_epoch( model, train_dataloader, dev_dataloader, optimizer, scheduler )
        print(stats)    
        
        if stats['accuracy'] > best_score :
            best_score = stats['accuracy']
            f = out_dir +'/' + 'best_model_ch.pt'
            torch.save({
                'epoch': epoch,
                'model': model,
                'stats' : stats,
                }, f)
        
        stats_vec.append(stats)
    
    stats_vec = pd.DataFrame(stats_vec)
    
    f = out_dir +'/' + 'last_model_ch.pt'
    torch.save({
        'epoch': epoch,
        'model': model,
        'stats' : stats,
        }, f)
    
    
    print(stats_vec)
    stats_vec.to_csv(out_dir +'/' + 'stats.csv' )

    




if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='run toxic bert model')
    parser.add_argument('-ptf', dest='pos_train_file')
    parser.add_argument('-ntf', dest='neg_train_file')
    parser.add_argument('-pdf', dest='pos_dev_file')
    parser.add_argument('-ndf', dest='neg_dev_file')
    
    parser.add_argument('-nrt', dest='nrows_train')
    parser.add_argument('-nrd', dest='nrows_dev')
    parser.add_argument('-ep', dest='epochs')
    
    parser.add_argument('-o', dest='out_dir')
    
    
    args = parser.parse_args()
    
    pos_train_file = args.pos_train_file
    neg_train_file = args.neg_train_file
    
    pos_dev_file = args.pos_dev_file
    neg_dev_file = args.neg_dev_file
    nrows_train = args.nrows_train
    nrows_dev = args.nrows_dev
    epochs = int(args.epochs)
    out_dir = args.out_dir
    
    try :
        nrows_train = int(nrows_train)
    except Exception as e :       
        nrows_train = None    
        
    try :
        nrows_dev = int(nrows_dev)
    except Exception as e :       
        nrows_dev = None    
        
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')   
    if torch.cuda.is_available():  
       dev = "cuda:0" 
    else:  
       dev = "cpu"  
    device = torch.device(dev) 
        
    run_model(pos_train_file , neg_train_file, pos_dev_file , neg_dev_file , nrows_train, nrows_dev, epochs , out_dir)    
