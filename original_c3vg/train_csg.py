from torch.utils.data import DataLoader
import random
import torch.optim as optim
import torch.nn as nn
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import json
import pickle
from model import Gen
from utils import Data_Process
import os
import re
def relaw(rel, text):
    p = re.compile(rel)
    m = p.search(text)
    if m is None:
        return None
    return m.group(0)

def re_view(xx):
    re_artrule = r'本院 认为(.*)其 行为'
    art = relaw(re_artrule, xx)
    if art == None:
        return -1
    return art[12:-4]
class Voc:
    def __init__(self, name):
        self.name = name
        self.word2id = {"SOS_token":2,"EOS_token":1,"PAD_token":0,'UNK':3}
        self.word2count = {}
        self.id2word = {2: "SOS_token", 1: "EOS_token", 0:"PAD_token",3:'UNK'}
        self.n_words = 4  # Count SOS and EOS
        self.symol = ['，','？','《','》','【','】','（','）','、','。','：','；']
    def addSentence(self, sentence):
        for word in sentence.strip().split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word=='':
            return
        if word in self.symol:
            return 
        if word not in self.word2id :
            self.word2id[word] = self.n_words
            self.word2count[word] = 1
            self.id2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1  
        
random.seed(42)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    )
logger = logging.getLogger(__name__)

SOS_token = 2
EOS_token = 1
data_all = []
random.seed(42)
f = open('./train.json')
for index,line in enumerate(f):
    # if index>10:
    #     break
    lines = json.loads(line)
    ll = len(lines['fact'].split('。'))
    if re_view(lines['adc']) == -1:
        continue
    charge = lines['charge']
    if charge=='其他刑事犯':
        continue
    data_all.append(line)

random.shuffle(data_all)
print(len(data_all))
dataloader = DataLoader(data_all, batch_size=128, shuffle=False, num_workers=0, drop_last=False)
criterion = nn.CrossEntropyLoss()
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载词表
with open('./encoder_vocab.pickle', 'rb') as file:
    encoder_vocab=pickle.load(file)
    
with open('./decoder_vocab.pickle', 'rb') as file:
    decoder_vocab=pickle.load(file)
        
src_vocab = len(encoder_vocab.word2id)
tgt_vocab = len(decoder_vocab.word2id)
process = Data_Process(encoder_vocab,decoder_vocab)
model = Gen(device,src_vocab, tgt_vocab)
model = model.to(device)


learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epoch = 16
global_step = 0

for epoch in range(num_epoch):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    logger.info("Trianing Epoch: {}/{}".format(epoch+1, int(num_epoch)))
    
    for step,batch in enumerate(tqdm(dataloader)):
    
        global_step += 1
        nb_tr_steps += 1
        model.train()
        optimizer.zero_grad()

        charge_label,sc_source,sc_target,adc_source,adc_target,_ = process.process_data_c3g(batch)
        
        sc_source = sc_source.to(device)
        adc_source = adc_source.to(device)
        sc_target = sc_target.to(device)
        adc_target = adc_target.to(device)
        charge_label = charge_label.to(device)
    
        loss_sc,loss_adc,charge_out = model(sc_source,adc_source,sc_target,adc_target,training=True)

        loss_charge = criterion(charge_out,charge_label)

        loss = loss_sc+loss_adc+loss_charge
        tr_loss+=loss.item()
        loss.backward()
        optimizer.step()

        if global_step%1000 == 0:
            logger.info("Training loss: {}, global step: {}".format(tr_loss/nb_tr_steps, global_step))


    PATH = './output/'+'_c3vg_'+str(epoch)
    torch.save(model.state_dict(), PATH)
