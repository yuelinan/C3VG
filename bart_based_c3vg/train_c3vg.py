import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob 
import json
from sklearn import metrics
import numpy as np
import torch
import os
import torch.nn as nn
from utils import *
from model import *
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import prettytable as pt
from sacrebleu.metrics import BLEU

parser = argparse.ArgumentParser(description='VMask classificer')
# batch 128, gpu 10000M
parser.add_argument('--aspect', type=int, default=0, help='aspect')
parser.add_argument('--dataset', type=str, default='beer')
parser.add_argument('--model_name', type=str, default='RNP', help='model name')
parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
parser.add_argument('--max_len', type=int, default=300, help='max_len')
parser.add_argument('--weight_decay', default=0, type=float, help='adding l2 regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='the probability for dropout')
parser.add_argument('--alpha_rationle', type=float, default=0.2, help='alpha_rationle')
parser.add_argument('--hidden_dim', type=int, default=768, help='number of hidden dimension')
parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
        non-linearity transfer function')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--types', type=str, default='legal', help='data_type')
parser.add_argument('--epochs', type=int, default=16, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--class_num', type=int, default=2, help='class_num')
parser.add_argument('--save_path', type=str, default='', help='save_path')
parser.add_argument('--max_target_length', type=int, default=15, help='save_path')


args = parser.parse_args()

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    )
logger = logging.getLogger(__name__)


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

def random_seed():
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.device)
process = Data_Process(args)

id2charge = json.load(open('./id2charge.json'))

for k, v in vars(args).items():
    logger.info("{:20} : {:10}".format(k, str(v)))

def get_charge_name(text):
    text_list = re.findall(".{1}",text)
    new_text = " ".join(text_list)
    return new_text+' 罪。'
def main():
    model = eval(args.model_name)(args)
    data_all = process.process_data(args.types,model.sc_gen.model)
    print(len(data_all))
    dataloader = DataLoader(data_all, batch_size = args.batch_size, shuffle=True, num_workers=0, drop_last=False)

    test_data_all = process.process_data('test',model.sc_gen.model)
    print(len(test_data_all))
    test_dataloader = DataLoader(test_data_all, batch_size = args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr)   

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(data_all) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    model.train()

    for epoch in range(1, args.epochs+1):
        model.train()

        logger.info("Trianing Epoch: {}/{}".format(epoch, int(args.epochs)))
        for step,batch in enumerate(tqdm(dataloader)):
            # if step>2:break
            batch = tuple(t.to(args.device) for t in batch) 
            
            adc_source_input_ids,adc_source_attention_mask, adc_target_input_ids,adc_decoder_input_ids, sc_source_input_ids,sc_source_attention_mask, sc_target_input_ids,sc_decoder_input_ids, charge_label, view_all_input_ids = batch
        
            optimizer.zero_grad()
            
            sc_output,adc_output,charge_out = model(adc_source_input_ids,adc_source_attention_mask, adc_target_input_ids,adc_decoder_input_ids, sc_source_input_ids,sc_source_attention_mask, sc_target_input_ids,sc_decoder_input_ids)

            loss = sc_output.loss + adc_output.loss
            loss += criterion(charge_out,charge_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()


        if epoch>0:
            model.eval()
            bleu = BLEU()

            ##############################  test  ##############################
            preds, candicate = [], []
            predictions_charge = []
            true_charge = []
            for step,batch in enumerate(tqdm(test_dataloader)):
                batch = tuple(t.to(args.device) for t in batch) 
                adc_source_input_ids,adc_source_attention_mask, adc_target_input_ids,adc_decoder_input_ids, sc_source_input_ids,sc_source_attention_mask, sc_target_input_ids,sc_decoder_input_ids, charge_label, view_all_input_ids = batch
                true_charge.extend(charge_label.cpu().numpy())
                with torch.no_grad():
                    # decode sc
                    # decode adc and charge
                    _,_,charge_out = model(adc_source_input_ids,adc_source_attention_mask, adc_target_input_ids,adc_decoder_input_ids, sc_source_input_ids,sc_source_attention_mask, sc_target_input_ids,sc_decoder_input_ids)

                    adc_generated_tokens = model.adc_gen.model.generate(adc_source_input_ids,max_length=50,num_beams=5).cpu().numpy()
                    sc_generated_tokens = model.sc_gen.model.generate(sc_source_input_ids,max_length=50,num_beams=5).cpu().numpy()
                
                charge_pred = charge_out.cpu().argmax(dim=1).numpy()
                predictions_charge.extend(charge_pred)
                
                charge_name = [get_charge_name(id2charge[str(i)]) for i in charge_pred]

                label_tokens = view_all_input_ids.cpu().numpy()   

                adc_decoded_preds = process.tokenizer.batch_decode(adc_generated_tokens, skip_special_tokens=True)
                sc_decoded_preds = process.tokenizer.batch_decode(sc_generated_tokens, skip_special_tokens=True)
                
                decoded_preds = ['本 院 认 为 被 告 人 '+ adc_decoded_preds[i] + ' 其 行 为 已 构 成 ' + charge_name[i] + ' '+ sc_decoded_preds[i]  for i in range(len(adc_decoded_preds))]
                
                # label_tokens = np.where(label_tokens != -100, label_tokens, process.tokenizer.pad_token_id)
                decoded_labels = process.tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
                preds += [pred.strip() for pred in decoded_preds]
                candicate += [[label.strip()] for label in decoded_labels]
                
            # bleu_score = bleu.corpus_score(preds, candicate).score
            if epoch>0:
                path = './output/result' + str(epoch) + '.json'
                f_result = open(path,'a+')
                result = {}
                result['preds'] = preds
                result['candicate'] = candicate
                json_str = json.dumps(result, ensure_ascii=False)
                f_result.write(json_str)
                f_result.write('\n')

            # logger.info(f"test_BLEU: {bleu_score:>0.2f}\n")
            bleu_score1 = bleu.corpus_score(preds, candicate,weights=(1., 0., 0., 0.)).score
            bleu_score2 = bleu.corpus_score(preds, candicate,weights=(0, 1, 0, 0)).score
            bleu_score3 = bleu.corpus_score(preds, candicate,weights=(0, 0, 1, 0)).score
            bleu_score4 = bleu.corpus_score(preds, candicate,weights=(0, 0, 0, 1)).score
            logger.info(f"test_BLEU1: {bleu_score1:>0.2f}\n")
            logger.info(f"test_BLEU2: {bleu_score2:>0.2f}\n")
            logger.info(f"test_BLEU3: {bleu_score3:>0.2f}\n")
            logger.info(f"test_BLEU4: {bleu_score4:>0.2f}\n")

            
            class_micro_f1, class_macro_precision, class_macro_recall, class_macro_f1 = eval_data_types(true_charge,predictions_charge,num_labels=62)
            table = pt.PrettyTable(['types ','     Acc   ','          P          ', '          R          ', '      F1          ' ]) 
            table.add_row(['dev',  class_micro_f1, class_macro_precision, class_macro_recall, class_macro_f1 ])
            logger.info(table)
        # save the last model
        if epoch >0:
            PATH = args.save_path+args.dataset+'_'+args.model_name.lower()+'_'+str(epoch)
            torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    random_seed()
    criterion = nn.CrossEntropyLoss()
    main()
