#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:17:43 2020

@author: raghuramkowdeed
"""


import argparse

from bert_toxic import BertForToxic
#from distilbert_toxic import DistilBertForToxic
from ganbert_toxic import DistilBertForToxic
#from large_bert_toxic import DistilBertForToxic
from data_utils import get_data_loader_bal, get_device, get_data_pred, get_data_loader_pred

import numpy as np
import torch
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from transformers import get_linear_schedule_with_warmup

from transformers import BertModel, BertConfig, AdamW, DistilBertConfig
import os 

from transformers import BertTokenizer, DistilBertTokenizer

def train_epoch( model, train_dataloader, dev_dataloader, optimizer, scheduler ):
    
    ## For each batch, we must reset the gradients
    ## stored by the model.   
    train_loss = 0
    train_loss_y = 0
    train_loss_z = 0
    count = 0    
    for train_data in tqdm( train_dataloader ):
        
        try :
        
        # clear gradients
            optimizer.zero_grad()
            # evoke model in training mode on batch
            model.train()
            prob, loss, y_loss, z_loss = model.forward(input_ids=train_data[0],attention_mask=train_data[1],  target = train_data[2],
                                       sample_weights = train_data[3], mode = 'train',  idn_score = train_data[4])
            
            # compute loss w.r.t batch
            
            # pass gradients back, startiing on loss value
            loss.backward()
    
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
            # update parameters
            optimizer.step()
    
            scheduler.step()
            
            train_loss += loss.item()
            train_loss_y += y_loss.item()
            train_loss_z += z_loss.item()
            count = count + 1
            
        except Exception as e :
            print(e)
            print(train_data[-1])
            print(train_data[-2])
            raise
            

    train_loss = train_loss/count
    train_loss_y = train_loss_y/count
    train_loss_z = train_loss_z/count

    pred_labels = []
    target_labels = []
    probs = []
    dev_loss = 0.0
    dev_loss_y = 0.0
    dev_loss_z = 0.0
    thres = 0.5
    count = 0 
    txts = []
    for dev_data in tqdm( dev_dataloader ) :
       model.eval()

       with torch.no_grad() :
         
         prob, loss, y_loss, z_loss = model.forward(input_ids=dev_data[0],attention_mask=dev_data[1],  target = dev_data[2],
                                   sample_weights = dev_data[3], mode='eval', idn_score = dev_data[4])
         
         
         dev_loss += loss.item()
         dev_loss_y += y_loss.item()
         dev_loss_z += z_loss.item()
         
         count = count + 1


         prob = prob.cpu().detach().numpy() 
         prob[prob<thres] = 0
         prob[prob>=thres] = 1
         pred_labels.append(prob)
         target = dev_data[2].cpu().detach().numpy() 
         
         target_labels.append(target)
         txts.append(dev_data[-2])

    dev_loss = dev_loss / count 
    dev_loss_y = dev_loss_y / count 
    dev_loss_z = dev_loss_z / count 
    
    target_labels = np.concatenate(target_labels)
    pred_labels = np.concatenate(pred_labels)
    txts = np.concatenate(txts)
    
    dev_pred = pd.DataFrame()
    dev_pred['txt'] = txts
    dev_pred['pred'] = pred_labels
    dev_pred['target'] = target_labels
    
    target_labels[target_labels<thres] = 0
    target_labels[target_labels>=thres] = 1
    
    
    accuracy = accuracy_score(target_labels, pred_labels, )
    f1 = f1_score(target_labels, pred_labels, )
    
    precision = precision_score(target_labels, pred_labels, )
    recall = recall_score(target_labels, pred_labels, )
    
    stats = {'accuracy': accuracy, 'precision':precision, 'recall':recall, 'f1':f1, 'dev_loss':dev_loss, 'train_loss':train_loss,  
             'dev_loss_y':dev_loss_y, 'train_loss_y':train_loss_y, 'dev_loss_z':dev_loss_z, 'train_loss_z':train_loss_z,}
    
    

    
    # return the total to keep track of how you did this time around
    return stats, dev_pred



def run_model(pos_train_file , neg_train_file, pos_dev_file , neg_dev_file , nrows_train, 
              nrows_dev, epochs , out_dir, dropout = 0.2, model = 'bert', batch_size = 16, 
              test_file = '../data/test_data_clean.csv', lr = 2e-5, lmda = 10.0, stnc_emb = 'last') :
    
    
    
    device = get_device()
    
    bert_hidden_states = 4
    
    
    
    if model == 'bert' :
       config = BertConfig()
       config.output_hidden_states = True
       model = BertForToxic(config,  bert_hidden_states=bert_hidden_states, dropout = dropout, update_bert = True, )       
       tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    if model == 'distilbert' :
        #config = DistilBertConfig()
        config = BertConfig()
        config.output_hidden_states = True
        model = DistilBertForToxic(config,  bert_hidden_states=bert_hidden_states, dropout = dropout, 
                                   update_bert = True, lmda= lmda,stnc_emb= stnc_emb)
        #tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_dataloader = get_data_loader_bal(pos_train_file, neg_train_file, batch_size=batch_size, nrows_pos= nrows_train, nrows_neg = nrows_train*10, mode = 'train', tokenizer=tokenizer)
    dev_dataloader =   get_data_loader_bal(pos_dev_file, neg_dev_file, batch_size=batch_size, nrows_pos= nrows_dev, nrows_neg = nrows_dev, mode = 'dev', tokenizer=tokenizer)
        
        
        
    
    model.to(device)
    
    optimizer = AdamW(model.parameters(),
                      lr = lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
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
    dev_pred_vec = []
    for epoch in range(epochs):
        stats, dev_pred = train_epoch( model, train_dataloader, dev_dataloader, optimizer, scheduler )
        print(epoch,stats)    
        
        if stats['accuracy'] > best_score :
            best_score = stats['accuracy']
            f = out_dir +'/' + 'best_model_ch.pt'
            torch.save({
                'epoch': epoch,
                'model': model,
                'stats' : stats,
                }, f)
        
        stats_vec.append(stats)
        dev_pred_vec.append(dev_pred)
    
    stats_vec = pd.DataFrame(stats_vec)
    dev_pred_vec = pd.concat(dev_pred_vec, axis=0)
    
    f = out_dir +'/' + 'last_model_ch.pt'
    torch.save({
        'epoch': epoch,
        'model': model,
        'stats' : stats,
        }, f)
    
    
    print(stats_vec)
    stats_vec.to_csv(out_dir +'/' + 'stats.csv' )
    

    out_file = out_dir + '/train_pred.csv'   
    df = get_data_pred(train_dataloader, model,  out_file, )
    
    
    out_file = out_dir + '/dev_pred.csv'   
    df = get_data_pred(dev_dataloader, model,  out_file, )
 
    
    test_dataloader = get_data_loader_pred(test_file, tokenizer, nrows=None)
    out_file = out_dir + '/test_pred.csv'   
    df = get_data_pred(test_dataloader, model,  out_file, )
    
    
    
    
    

    


'''

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='run toxic bert model')
    parser.add_argument('-ptf', dest='pos_train_file')
    parser.add_argument('-ntf', dest='neg_train_file')
    parser.add_argument('-pdf', dest='pos_dev_file')
    parser.add_argument('-ndf', dest='neg_dev_file')
    
    parser.add_argument('-nr', dest='nrows')
    parser.add_argument('-ep', dest='epochs')
    
    parser.add_argument('-o', dest='out_dir')
    
    
    args = parser.parse_args()
    
    pos_train_file = args.pos_train_file
    neg_train_file = args.neg_train_file
    
    pos_dev_file = args.pos_dev_file
    neg_dev_file = args.neg_dev_file
    nrows = args.nrows
    epochs = int(args.epochs)
    out_dir = args.out_dir
    
    try :
        nrows = int(nrows)
    except Exception as e :       
        nrows = None    
        
    run_model(pos_train_file , neg_train_file, pos_dev_file , neg_dev_file , nrows, epochs , out_dir)    

   ''' 
