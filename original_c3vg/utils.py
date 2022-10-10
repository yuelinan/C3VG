import json
import torch
import re
import numpy as np
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

class Data_Process():
    def __init__(self,encoder_vocab,decoder_vocab):
        self.symbol = [",", ".", "?", "\"", "”", "。", "？", "","，",",","、","”",'“','；']
        self.word2id = encoder_vocab.word2id
        self.word2id4dec = decoder_vocab.word2id
        self.last_symbol = ["。"]
        self.time2id = json.load(open('./time2id.json'))
        self.id2term_test = json.load(open('./id2term_test.json'))
        self.charge2id = json.load(open('./charge2id.json'))
        
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
            if word in self.symbol:
                continue
            result.append(word)
        return result
    def parse_view(self,sent):
        result = []
        sent = sent.strip().split()
        for word in sent:
            if len(word) == 0:
                continue
            if word in self.symbol:
                continue
            result.append(word)
        view = ''
        for i in result:
            view+=i
            view+=' '
        return view[:-1]
    def parseH(self, sent):
        result = []
        temp = []     
        sent = sent.strip().split()
        for word in sent:
            if word in self.symbol and word not in self.last_symbol:
                continue
            temp.append(word)
            last = False
            for symbols in self.last_symbol:
                if word == symbols:
                    last = True
            if last:
                #不要标点
                result.append(temp[:-1])
                temp = []
        if len(temp) != 0:
            result.append(temp)
        
        return result

    def seq2Htensor(self, docs, max_sent=16, max_sent_len=128):
        
        sent_num_max = max([len(s) for s in docs])

        sent_len_max = max([len(w) for s in docs for w in s])
        sent_len_max = min(sent_len_max, max_sent_len)
        # for lstm encoder
        sent_tensor = torch.LongTensor(len(docs), sent_num_max, sent_len_max).zero_()
        
        sent_len = torch.LongTensor(len(docs), sent_num_max).zero_()
        #doc_len = torch.LongTensor(len(docs)).zero_()
        for d_id, doc in enumerate(docs):
            #doc_len[d_id] = len(doc)
            for s_id, sent in enumerate(doc):
                if s_id >= sent_num_max: break
                sent_len[d_id][s_id] = len(sent)
                for w_id, word in enumerate(sent):
                    if w_id >= sent_len_max: break
                    sent_tensor[d_id][s_id][w_id] = self.transform(word)
                    
        return sent_tensor,sent_len,sent_num_max

    def seq2tensor(self, sents, types ,max_len):
        #print('aaa')

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


    def process_data(self,data):

        source = []
        adc_target = []
        sc_target = []
        sc_source = []
        adc_source = []
        charge_label = [] 
        for index,line in enumerate(data):
            sc = ''
            adc = ''            
            line = json.loads(line)                  
            fact = line['fact'].split('。')
            label = [0 for i in range(len(fact))]
            for i in line['label']:
                label[i] = 1
            time = self.id2term_test[line['id']]
            time_labels = self.time2id[str(time)]
            for index,num in enumerate(label):
                if num==1:
                    sc+=fact[index]
                    sc+= ' '
                else:
                    adc+=fact[index]
                    adc+= ' '
            if adc == '' or sc =='':
                continue
            
            adc_source.append(self.parse(adc)) 
            sc_source.append(self.parse(sc))
            
            adc_target.append(self.parse(re_view(line['adc'])))
            sc = ''
            for i in line['sc']:
                sc+=i
                sc+=' '
            
            sc_target.append(self.parse((sc)))
  
            view = '本院 认为 ， '+re_view(line['adc'])+ ' 其 行为已 构成 '+ line['charge']+' 罪'

            for sen in line['sc']:
                view+=' '
                view+=sen

            charge = line['charge']
            charge_label.append(self.charge2id[charge])
        charge_label = torch.tensor(charge_label,dtype=torch.long)

        sc_source,_ = self.seq2tensor(sc_source,types = 'encoder', max_len=200)
        adc_source,_ = self.seq2tensor(adc_source, types = 'encoder', max_len=300)
        adc_target,_ = self.seq2tensor(adc_target, types = 'decoder', max_len=150)
        sc_target,_ = self.seq2tensor(sc_target, types = 'decoder', max_len=200)

        return charge_label,sc_source,sc_target,adc_source,adc_target,self.parse_view(view),time_labels
    

    def process_data_c3g(self,data):

        source = []
        adc_target = []
        sc_target = []
        sc_source = []
        adc_source = []
        charge_label = [] 
        for index,line in enumerate(data):
            sc = ''
            adc = ''            
            line = json.loads(line)                  
            fact = line['fact'].split('。')
            label = [int(i) for i in line['result_label']]

            time = self.id2term_test[line['id']]
            time_labels = self.time2id[str(time)]
                
            for index,num in enumerate(label):
                if num==1:
                    sc+=fact[index]
                    sc+= ' '
                else:
                    adc+=fact[index]
                    adc+= ' '
            if adc == '' or sc =='':
                continue
            
            adc_source.append(self.parse(adc)) 
            sc_source.append(self.parse(sc))
            
            adc_target.append(self.parse(re_view(line['adc'])))
            sc = ''
            for i in line['sc']:
                sc+=i
                sc+=' '
            
            sc_target.append(self.parse((sc)))
  
            view = '本院 认为 ， '+re_view(line['adc'])+ ' 其 行为已 构成 '+ line['charge']+' 罪'

            for sen in line['sc']:
                view+=' '
                view+=sen

            charge = line['charge']
            charge_label.append(self.charge2id[charge])
        charge_label = torch.tensor(charge_label,dtype=torch.long)

        sc_source,_ = self.seq2tensor(sc_source,types = 'encoder', max_len=200)
        adc_source,_ = self.seq2tensor(adc_source, types = 'encoder', max_len=300)
        adc_target,_ = self.seq2tensor(adc_target, types = 'decoder', max_len=150)
        sc_target,_ = self.seq2tensor(sc_target, types = 'decoder', max_len=200)

        return charge_label,sc_source,sc_target,adc_source,adc_target,self.parse_view(view),time_labels
    

