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
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
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

    logger.info("Micro precision: {}".format(micro_precision))
    logger.info("Micro recall: {}".format(micro_recall))
    logger.info("Micro f1: {}".format(micro_f1))
    logger.info("Macro precision: {}".format(macro_precision))
    logger.info("Macro recall: {}".format(macro_recall))
    logger.info("Macro f1: {}".format(macro_f1))

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

SOS_token = 2
EOS_token = 1
data_all = []
random.seed(42)
f = open('./test.json')
for index,line in enumerate(f):
    lines = json.loads(line)
    ll = len(lines['fact'].split('。'))
    if re_view(lines['adc']) == -1:
        continue
    charge = lines['charge']
    if charge=='其他刑事犯':
        continue
    
    label = [int(i) for i in lines['result_label']]

    if np.mean(label)==1:
     
        continue

    data_all.append(line)

print(len(data_all))

id2charge = json.load(open('./id2charge.json'))
dataloader = DataLoader(data_all, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
#print(train_data)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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

model_name = '_c3vg_15'
PATH = './output/'+model_name
model.load_state_dict(torch.load(PATH))
model.to(device)

true_charge = []
predictions_charge = []
rouge = Rouge()
x1 = []
x2 = []
xl = []

b1 = []
b2 = []
b3 = []
b4 = []
s = 0
for step,batch in enumerate(tqdm(dataloader)):
    model.eval()
    
    charge_label,sc_source,sc_target,adc_source,adc_target,view,time_labels = process.process_data_c3g(batch)
    true_charge.extend(charge_label.numpy())
    sc_source = sc_source.to(device)
    adc_source = adc_source.to(device)
    sc_target = sc_target.to(device)
    adc_target = adc_target.to(device)
    
    with torch.no_grad():
        result1,result2,charge_out = model(sc_source,adc_source,sc_target,adc_target,training=False)

    charge_pred = charge_out.cpu().argmax(dim=1).numpy()
    predictions_charge.extend(charge_pred)
    result_seq = []
    view_list = []
    result_sentence = '本院 认为 被告人 '

    for i in result2[0][0]:
        if i == 'PAD_token':
            continue
        if i=='UNK':
            continue
        if i=='<EOS>':
            continue
        result_sentence+=i
        result_sentence+=' '
    result_sentence+=' 其 行为已 构成 '
    result_sentence+= id2charge[str(charge_pred[0])]
    result_sentence+=' 罪 '


    for i in result1[0][0]:
        if i == 'PAD_token':
            continue
        if i=='UNK':
            continue
        if i=='<EOS>':
            continue
        result_sentence+=i
        result_sentence+=' '

    if result_sentence != ''  and view!='' and len(view)<2000:
        result_seq.append(result_sentence)
        view_list.append(view)
        s+=1

        rouge_score = rouge.get_scores(result_seq, view_list)


        for i in range(len(result_seq)):
            x1.append(rouge_score[i]["rouge-1"]['f'])
            x2.append(rouge_score[i]["rouge-2"]['f'])
            xl.append(rouge_score[i]["rouge-l"]['f'])

        reference = [view.split(' ')]
        candidate = result_sentence.split(' ')
        b1.append(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
        b2.append(sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
        b3.append(sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
        b4.append(sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))



logger.info("ruoge1: {}".format(np.mean(x1)))
logger.info("ruoge2: {}".format(np.mean(x2)))
logger.info("ruogeL: {}".format(np.mean(xl)))


logger.info("BLEU1: {}".format(np.mean(b1)))
logger.info("BLEU2: {}".format(np.mean(b2)))
logger.info("BLEU3: {}".format(np.mean(b3)))
logger.info("BLEU4: {}".format(np.mean(b4)))
logger.info("BLEUN: {}".format(np.mean([np.mean(b1),np.mean(b2),np.mean(b3),np.mean(b4)])))

print('ruoge1: %f' % np.mean(x1))
print('ruoge2: %f' % np.mean(x2))
print('ruogeL: %f' % np.mean(xl))

print('BLEU1: %f' % np.mean(b1))
print('BLEU2: %f' % np.mean(b2))
print('BLEU3: %f' % np.mean(b3))
print('BLEU4: %f' % np.mean(b4))


print('罪名')
eval_data_types(true_charge,predictions_charge,num_labels=62)
