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
from model import Seq2seqAttn
from utils import Data_Pre
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
random.seed(42)

def get_value(res):
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def gen_result(res, test=False, file_path=None, class_name=None):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_value(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_value(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    print("Micro precision\t%.4f" % micro_precision)
    print("Micro recall\t%.4f" % micro_recall)
    print("Micro f1\t%.4f" % micro_f1)
    print("Macro precision\t%.4f" % macro_precision)
    print("Macro recall\t%.4f" % macro_recall) 
    print("Macro f1\t%.4f" % macro_f1)

    return

def eval_data_types(target,prediction,num_labels):
    ground_truth_v2 = []
    predictions_v2 = []
    for i in target:
        v = [0 for j in range(num_labels)]
        v[i] = 1
        ground_truth_v2.append(v)
    for i in prediction:
        v = [0 for j in range(num_labels)]
        v[i] = 1
        predictions_v2.append(v)

    res = []
    for i in range(num_labels):
        res.append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    y_true = np.array(ground_truth_v2)
    y_pred = np.array(predictions_v2)
    for i in range(num_labels):
    
        outputs1 = y_pred[:, i]
        labels1 = y_true[:, i] 
        res[i]["TP"] += int((labels1 * outputs1).sum())
        res[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
        res[i]["FP"] += int(((1 - labels1) * outputs1).sum())
        res[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())
    gen_result(res)

    return 0

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

def beam_decode(decoder, decoder_hidden,decoder_vocab, encoder_outputs, beam_size, max_length=20):

    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden,decoder_vocab))
    for i in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = torch.LongTensor([[sentence.last_idx]])
            decoder_input = decoder_input.to(device)

            decoder_hidden = sentence.decoder_hidden
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(beam_size)
            term, top = sentence.addTopk(topi, topv, decoder_hidden, beam_size)
            terminal_sentences.extend(term)
            next_top_sentences.extend(top)

        next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
        prev_top_sentences = next_top_sentences[:beam_size]
        next_top_sentences = []

    terminal_sentences += [sentence.toWordScore() for sentence in prev_top_sentences]
    terminal_sentences.sort(key=lambda x: x[1], reverse=True)

    n = min(len(terminal_sentences), 15)
    return terminal_sentences[:n]

SOS_token = 2
EOS_token = 1
data_all = []

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('encoder_vocab.pickle', 'rb') as file:
    encoder_vocab=pickle.load(file)
    
with open('decoder_vocab.pickle', 'rb') as file:
    decoder_vocab=pickle.load(file)
        
src_vocab = len(encoder_vocab.word2id)
tgt_vocab = len(decoder_vocab.word2id)
process = Data_Pre(encoder_vocab,decoder_vocab)
model = Seq2seqAttn(src_vocab, tgt_vocab, device ,process)
model = model.to(device)

model_name = '_model0_15'
PATH = model_name
model.load_state_dict(torch.load(PATH))
model.to(device)

f = open('test.json')
for index,line in enumerate(f):

    lines = json.loads(line)
      
    if re_view(lines['adc']) == -1:
        continue
    charge = lines['charge']
    if charge=='其他刑事犯':
        continue
    data_all.append(line)

random.shuffle(data_all)
test_data = data_all
print(test_data)
dataloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0, drop_last=False)

result_seq = []
predictions_charge = []
true_charge = []
for step,batch in enumerate(tqdm(dataloader)):

    model.eval()
    source,target,charge_label = process.process(batch)
    true_charge.extend(charge_label.numpy())
    source = source.to(device)
    target = target.to(device)
    with torch.no_grad():
        results,charge_out = model(source,target,training=False)
    charge_pred = charge_out.cpu().argmax(dim=1).numpy()
    predictions_charge.extend(charge_pred)

    for data in results:
        result_sentence = ''
        for i in data[0]:
            result_sentence+=i
            result_sentence+=' '
        result_seq.append(result_sentence)
print(result_seq)
eval_data_types(true_charge,predictions_charge,num_labels=62)
