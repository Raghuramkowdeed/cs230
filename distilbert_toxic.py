#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:26:36 2020

@author: raghuramkowdeed
"""

from transformers import DistilBertForSequenceClassification, DistilBertPreTrainedModel, DistilBertModel
from copy import deepcopy
from data_utils import  get_device
from torch import nn
import numpy as np

import torch
from transformers import DistilBertModel, DistilBertConfig

class DistilBertForToxic(DistilBertPreTrainedModel):
    def __init__(self, config, bert_hidden_states=1, dropout = 0.1, update_bert = False):
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
        mode = 'train',
    ):
        
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
        
        
        for i in range(self.bert_hidden_states) :
            sequence_output.append( outputs[1][-i][:,0])  
            
        
        
        sequence_output = torch.cat( sequence_output ,dim=-1)
 
        
        prob = self.qa_outputs(sequence_output)        
        prob = prob.squeeze(-1)
        
        
        
        if target is not None :
             target = target.type(torch.float32)
             #print(prob.type(), target.type())
             #print('prob',prob, target)
             loss_f = nn.BCELoss()
             loss = loss_f(prob, target)
            
             return prob , loss
            
        
        

        return prob  # (loss), start_logits, end_logits, (hidden_states), (attentions)
