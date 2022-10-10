import json
import torch
import re
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader

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

def make_kv_string(d):
    out = []
    for k, v in d.items():
        if isinstance(v, float):
            out.append("{} {:.4f}".format(k, v))
        else:
            out.append("{} {}".format(k, v))

    return " ".join(out)

def get_value(res):
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

    #print("Micro precision\t%.4f" % micro_precision)
    #print("Micro recall\t%.4f" % micro_recall)
    print("Micro f1\t%.4f" % micro_f1)
    print("Macro precision\t%.4f" % macro_precision)
    print("Macro recall\t%.4f" % macro_recall) 
    print("Macro f1\t%.4f" % macro_f1)

    return micro_f1, macro_precision, macro_recall,macro_f1

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
    micro_f1, macro_precision, macro_recall,macro_f1 = gen_result(res)

    return micro_f1, macro_precision, macro_recall,macro_f1



class Data_Process():
    def __init__(self,args):
        
        self.charge2id = json.load(open('./charge2id.json'))
        self.tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
        self.train_data_path = './train.json'
        self.test_data_path = './test.json'
        self.max_len = args.max_len
    def encode_fn(self,text_list):	
        tokenizer = self.tokenizer.batch_encode_plus(
            text_list,
            padding = True,
            truncation = True,
            max_length = self.max_len,
            return_tensors='pt' 
        )
        input_ids = tokenizer['input_ids']
        attention_mask = tokenizer['attention_mask']
        return input_ids,attention_mask
    def process_data(self,types,model):
        if types == 'train':
            data_path = self.train_data_path
        else:
            data_path = self.test_data_path
        adc_target = []
        sc_target = []
        sc_source = []
        adc_source = []
        charge_label = [] 
        view_all = []
        f = open(data_path,'r',encoding='utf8')
        for index,line in enumerate(f):
            #print(line)
            if types == 'test':
               if index>1000:break
            #if index>20:break
            sc = ''
            adc = ''            
            line = json.loads(line)   
            crime_fact = ''.join(line['fact'].split())
            
            fact = line['fact'].split('。')
            label = [0 for i in range(len(fact))]
            for i in line['label']:
                label[i] = 1
            # extract the adc and sc
            for index,num in enumerate(label):
                if num==1:
                    sc+=fact[index]
                    sc+= '。'
                else:
                    adc+=fact[index]
                    adc+= '。'
            if adc == '' or sc =='':
                continue
            sc = ''.join(sc.split())
            adc = ''.join(adc.split())

            view = '本院 认为 ， '+re_view(line['adc'])+ ' 其 行为已 构成 '+ line['charge']+' 罪。'
            sc_view = ''
            for sen in line['sc']:
                sc_view+=sen
                sc_view+='。'
            view+=sc_view
            crime_views = ''.join(view.split())
            view_all.append(crime_views)
            adc_view = ''.join(re_view(line['adc']).split())
            sc_view = ''.join(sc_view.split())

            charge = line['charge']
            charge_label.append(self.charge2id[charge])

            adc_source.append(adc)
            adc_target.append(adc_view)

            sc_source.append(sc)
            sc_target.append(sc_view)
            
            
        charge_label = torch.tensor(charge_label,dtype=torch.long)
        adc_source_input_ids,adc_source_attention_mask = self.encode_fn(adc_source)
        adc_target_input_ids,_ = self.encode_fn(adc_target)
        adc_decoder_input_ids = model.prepare_decoder_input_ids_from_labels(adc_target_input_ids)


        sc_source_input_ids,sc_source_attention_mask = self.encode_fn(sc_source)
        sc_target_input_ids,_ = self.encode_fn(sc_target)
        sc_decoder_input_ids = model.prepare_decoder_input_ids_from_labels(sc_target_input_ids)

        view_all_input_ids,_ = self.encode_fn(view_all)
        data = TensorDataset(adc_source_input_ids,adc_source_attention_mask, adc_target_input_ids,adc_decoder_input_ids, sc_source_input_ids,sc_source_attention_mask, sc_target_input_ids,sc_decoder_input_ids, charge_label, view_all_input_ids)

        
        return data
    
