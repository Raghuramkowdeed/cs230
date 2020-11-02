#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:53:58 2020

@author: raghuramkowdeed
"""

import argparse
from run_toxic_bert import run_model

import random
import numpy as np
import torch

if __name__ == '__main__' :
    seed_val = 1125

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    
    parser = argparse.ArgumentParser(description='run toxic bert model')
    parser.add_argument('-ptf', dest='pos_train_file')
    parser.add_argument('-ntf', dest='neg_train_file')
    parser.add_argument('-pdf', dest='pos_dev_file')
    parser.add_argument('-ndf', dest='neg_dev_file')
    
    parser.add_argument('-nrt', dest='nrows_train')
    parser.add_argument('-nrd', dest='nrows_dev')
    parser.add_argument('-ep', dest='epochs')
    
    parser.add_argument('-o', dest='out_dir')
    parser.add_argument('-dp', dest='dropout', default= '0.2' )
    parser.add_argument('-m', dest='model', default= 'bert' )
    parser.add_argument('-tf', dest='test_file', default= '../data/test_data_clean.csv' )
    parser.add_argument('-lr', dest='learning_rate', default= 2e-5 )
    
    args = parser.parse_args()
    
    pos_train_file = args.pos_train_file
    neg_train_file = args.neg_train_file
    
    pos_dev_file = args.pos_dev_file
    neg_dev_file = args.neg_dev_file
    nrows_train = args.nrows_train
    nrows_dev = args.nrows_dev
    epochs = int(args.epochs)
    out_dir = args.out_dir
    dropout = float( args.dropout )
    model_name = args.model
    test_file = args.test_file
    learning_rate = float( args.learning_rate )
    
    try :
        nrows_train = int(nrows_train)
    except Exception as e :       
        nrows_train = None    
        
    try :
        nrows_dev = int(nrows_dev)
    except Exception as e :       
        nrows_dev = None    
        
    run_model(pos_train_file , neg_train_file, pos_dev_file , neg_dev_file , nrows_train, nrows_dev, 
              epochs , out_dir, dropout, model_name, test_file = test_file, lr = learning_rate )    
