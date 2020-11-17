#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:58:07 2020

@author: raghuramkowdeed
"""

from transformers import DistilBertForSequenceClassification, DistilBertPreTrainedModel, DistilBertModel
from copy import deepcopy
from data_utils import  get_device
from torch import nn
import numpy as np

import torch
from transformers import DistilBertModel, DistilBertConfig


class Gac(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    
    lmda = 1.0 

    @staticmethod
    def forward(ctx, input, ):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = Gac.lmda * grad_input * -1.0
        
        #print('lmda', Gac.lmda)
        
        return grad_input


class DistilBertForToxic(DistilBertPreTrainedModel):
    def __init__(self, config, bert_hidden_states=1, dropout = 0.1, update_bert = False, lmda = 1.0, stnc_emb = 'last'):
        config = deepcopy(config)
        config.output_hidden_states = True
        config.dropout = dropout
        super(DistilBertForToxic, self).__init__(config)
        
        
        self.bert_hidden_states = bert_hidden_states
        self.num_labels = 1
        self.update_bert = update_bert
        #bert=DistilBertModel(DistilBertConfig())
        
        
        bert = DistilBertForSequenceClassification.from_pretrained(
                   "distilbert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
                   num_labels = 2, # The number of output labels--2 for binary classification.
                                    # You can increase this for multi-class tasks.   
                    output_attentions = False, # Whether the model returns attentions weights.
                    output_hidden_states = True, # Whether the model returns all hidden-states.
                ).distilbert
        
        
        bert.config = config   
        device = get_device()
        bert = bert.to(device)
        self.bert = bert
        
        
        self.qa_outputs = nn.Sequential( nn.Dropout(dropout),
                                        nn.Linear(config.hidden_size*bert_hidden_states, 1),                                         
                                         nn.Sigmoid()
                                        
                      
                      
                      
                      )
        
        self.z_outputs = nn.Sequential( nn.Dropout(dropout),
                                        nn.Linear(config.hidden_size*bert_hidden_states, 1),                                         
                                         nn.Sigmoid()
                                        
                      
                      
                      
                      )
        
        Gac.lmda = lmda
        self.gac = Gac.apply
        
        self.stnc_emb = stnc_emb
        

        #self.init_weights()
        
        
        
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        target = None,
        sample_weights = None,
        idn_score = None,
        mode = 'train',
        
    ):
        
        stnc_emb = self.stnc_emb
        
        if self.update_bert and mode == 'train': 
            self.bert.train()
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                #token_type_ids=token_type_ids,
                #position_ids=position_ids,
                #head_mask=head_mask,
                #inputs_embeds=inputs_embeds,
            )
        else :
            with torch.no_grad() :
                self.bert.eval()
                outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    #token_type_ids=token_type_ids,
                    #position_ids=position_ids,
                    #head_mask=head_mask,
                    #inputs_embeds=inputs_embeds,
                )
                
                

        sequence_output= []       
        #sequence_output.append(  self.bert.pooler(outputs[0])  )
        
        
        for i in range(0, self.bert_hidden_states ) :
            if stnc_emb == 'last' :
               sequence_output.append( outputs[1][-i][:,0])  
            
            if stnc_emb == 'max' :
                sequence_output.append( outputs[1][-i])
                
            
        
        ##print('stnc emb', stnc_emb)
        #print('output shape', (outputs[1][-1]).shape)
        #print('seq shape', sequence_output[0].shape )
        sequence_output = torch.cat( sequence_output ,dim=-1)
        #print('seq shape', sequence_output.shape )
 
        
        prob = self.qa_outputs(sequence_output)        
        prob = prob.squeeze(-1)
        if stnc_emb == 'max' :
            prob,_ = torch.max(prob, dim=-1)
        
        #print('prob shape', prob.shape)
        
        z_prob = self.gac(sequence_output)        
        z_prob = self.z_outputs(z_prob)  
        z_prob = z_prob.squeeze(-1)
        if stnc_emb == 'max' :
            z_prob,_ = torch.max(z_prob, dim=-1)
        
        
        
        
        if target is not None :
             target = target.type(torch.float32)
             #print(prob.type(), target.type())
             #print('prob',prob, target)
             loss_f = nn.MSELoss()
             #nn.BCELoss()
             y_loss = loss_f(prob, target)
             #y_loss = y_loss / target.shape[0]
             
             
             z_score = idn_score.type(torch.float32)
             ind = (idn_score != -1).nonzero()
             ind = ind.squeeze(-1)
             #print(idn_score, ind)
             
             #print(ind.shape, z_prob.shape, z_score.shape)
             
             if ind.shape[0] > 0 :
                 z_prob = z_prob[ind]
                 z_score = z_score[ind]
                 
                 
                 z_loss = loss_f(z_prob, z_score)
                 #z_loss = z_loss / z_score.shape[0]
                 
                 loss = y_loss + z_loss
             else :
                 loss = y_loss
                 z_loss = torch.tensor(0)
            
             return prob , loss, y_loss, z_loss
            
        
        

        return prob  # (loss), start_logits, end_logits, (hidden_states), (attentions)
