from transformers import BartTokenizer, BartModel,BartForConditionalGeneration,BartPretrainedModel
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import datetime
from model_bart import BartForConditionalGeneration_for_adc
class SC_Generation(nn.Module):
    def __init__(self,args):
        super(SC_Generation, self).__init__()
        print('SC_Generation')
        self.model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese",return_dict=True)

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids):
        
        output = self.model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, decoder_input_ids = decoder_input_ids)
        
        return output 


class ADC_Generation(nn.Module):
    def __init__(self,args):
        super(ADC_Generation, self).__init__()
        print('ADC_Generation')
        self.model = BartForConditionalGeneration_for_adc.from_pretrained("fnlp/bart-base-chinese",return_dict=True)
        self.classifier = nn.Linear(768,62)
    def forward(self, input_ids, attention_mask, labels, decoder_input_ids):

        output = self.model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, decoder_input_ids = decoder_input_ids)
        last_hidden_state = self.model.last_hidden_state.mean(1)
        charge_out = self.classifier(last_hidden_state)
        return output ,charge_out

class Gen(nn.Module):
    def __init__(self,args):
        super(Gen, self).__init__()
        self.adc_gen = ADC_Generation(args)
        self.sc_gen = SC_Generation(args)
        

    def forward(self, adc_source_input_ids,adc_source_attention_mask, adc_target_input_ids,adc_decoder_input_ids, sc_source_input_ids,sc_source_attention_mask, sc_target_input_ids,sc_decoder_input_ids):
        # 默认batch为1
        sc_output = self.sc_gen(sc_source_input_ids,sc_source_attention_mask, sc_target_input_ids,sc_decoder_input_ids)
        adc_output,charge_out = self.adc_gen(adc_source_input_ids,adc_source_attention_mask, adc_target_input_ids,adc_decoder_input_ids)
        
        return sc_output,adc_output,charge_out