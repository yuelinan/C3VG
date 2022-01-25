import json
import torch
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

class Data_Pre():
    def __init__(self,encoder_vocab,decoder_vocab):
        super(Data_Pre, self).__init__()
        self.symol = [",", ".", "?", "\"", "”", "。", "？", "","，",",","、","”"]
        self.word2id = encoder_vocab.word2id
        self.word2id4dec = decoder_vocab.word2id
        self.charge2detail = json.load(open('charge_details.json','r'))
        self.charge2id = json.load(open('charge2id.json','r'))

    def transform(self, word,type='encoder'):
        if type=='encoder':
            if not (word in self.word2id.keys()):
                return self.word2id["UNK"]
            else:
                return self.word2id[word]
        else:
            if not (word in self.word2id4dec.keys()):
                return self.word2id4dec["UNK"]
            else:
                return self.word2id4dec[word]

    def parse(self, sent):
        result = []
        sent = sent.strip().split()
        for word in sent:
            if len(word) == 0:
                continue
            if word in self.symol:
                continue
            result.append(word)
        return result
    
    def seq2tensor(self, sents, types ,max_len):
        
        sent_len_max = max([len(s) for s in sents])
        sent_len_max = min(sent_len_max, max_len)
        EOS_token = 1  
        sent_tensor = torch.LongTensor(len(sents), sent_len_max+1).zero_()
        sent_len = torch.LongTensor(len(sents)).zero_()
        for s_id, sent in enumerate(sents):
            sent_len[s_id] = len(sent)
            for w_id, word in enumerate(sent):
                if w_id >= sent_len_max:
                    break
                sent_tensor[s_id][w_id] = self.transform(word,types) 
            if w_id==sent_len_max:
                sent_tensor[s_id][w_id] = EOS_token
            else:sent_tensor[s_id][w_id+1] = EOS_token
        return sent_tensor, sent_len
    
    def process(self,data):
        source = []
        target = []
        charge_label = []
        for index,line in enumerate(data):
            line = json.loads(line)
            
            charge = line['charge']
            
            charge_label.append(int(self.charge2id[charge]))
            source.append(self.parse(line['fact']))
            target.append(self.parse(re_view(line['adc'])))

        charge_label = torch.tensor(charge_label,dtype=torch.long)
        source,source_len = self.seq2tensor(source, types = 'encoder', max_len=500)
        target,target_len = self.seq2tensor(target, types = 'decoder', max_len=150)
        return source,target,charge_label
       
        
    def process_law(self,charge_names):

        legal = []
        for i in charge_names:   
            legal.append(self.parse(self.charge2detail[i]['定义']))
           
        legals,legals_len = self.seq2tensor(legal, types = 'encoder', max_len=100)

        return legals
