# -*- coding: utf-8 -*-
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
class Sentence:
    def __init__(self, decoder_hidden, decoder_vocab,last_idx=2, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores
        self.id2word = decoder_vocab.id2word

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        # return mean of sentence_score
        return sum(self.sentence_scores) / len(self.sentence_scores)

    def addTopk(self, topi, topv, decoder_hidden, beam_size,decoder_vocab):
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][0][i] == 1:
                terminates.append(([self.id2word[idx.item()] for idx in self.sentence_idxes] + ['<EOS>'],
                                   self.avgScore())) # tuple(word_list, score_float
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(topi[0][0][i])
            scores.append(topv[0][0][i])
            sentences.append(Sentence(decoder_hidden,decoder_vocab, topi[0][0][i], idxes, scores))
        return terminates, sentences

    def toWordScore(self):
        words = []
        EOS_token = 1
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<EOS>')
            else:
                words.append(self.id2word[self.sentence_idxes[i].item()])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<EOS>')
        return (words, self.avgScore())    
        
    
class EncRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, use_birnn, dout):
        super(EncRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed.weight.requires_grad = True
        self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers,batch_first=True,
                           bidirectional=use_birnn)
        self.dropout = nn.Dropout(dout)

    def forward(self, inputs):
        embs = self.dropout(self.embed(inputs))
        enc_outs, hidden = self.rnn(embs)
        return self.dropout(enc_outs), hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim, method):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim

        if method == 'general':
            self.w = nn.Linear(hidden_dim, hidden_dim)
        elif method == 'concat':
            self.w = nn.Linear(hidden_dim*2, hidden_dim)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_dim))

    def forward(self, dec_out, enc_outs):
        if self.method == 'dot':
            attn_energies = self.dot(dec_out, enc_outs)
        elif self.method == 'general':
            attn_energies = self.general(dec_out, enc_outs)
        elif self.method == 'concat':
            attn_energies = self.concat(dec_out, enc_outs)
        return F.softmax(attn_energies, dim=0)

    def dot(self, dec_out, enc_outs):
        return torch.sum(dec_out*enc_outs, dim=2)

    def general(self, dec_out, enc_outs):
        energy = self.w(enc_outs)
        return torch.sum(dec_out*energy, dim=2)

    def concat(self, dec_out, enc_outs):
        dec_out = dec_out.expand(enc_outs.shape[0], -1, -1)
        energy = torch.cat((dec_out, enc_outs), 2)
        return torch.sum(self.v * self.w(energy).tanh(), dim=2)

class Attn(nn.Module):
    def __init__(self, method='dot', hidden_size=150):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
    def forward(self, hidden, encoder_outputs):
        if self.method=='dot':
            attn_energies = self.score(hidden,encoder_outputs) 
            return torch.nn.functional.softmax(attn_energies, dim=-1)

    def score(self, hidden, encoder_output):
        energy = torch.bmm(hidden, encoder_output.transpose(1, 2))
        return energy

class DecRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, use_birnn, 
                 dout, attn):
        super(DecRNN, self).__init__()
        hidden_dim = hidden_dim*2 if use_birnn else hidden_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed.weight.requires_grad = True
        self.rnn = nn.GRU(embed_dim, hidden_dim , n_layers,batch_first=True)

        self.w = nn.Linear(hidden_dim*2, hidden_dim)
        self.attn = Attn('dot', hidden_dim)

        self.prediction = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(dout)

    def forward(self, inputs, hidden, enc_outs):
        #inputs = inputs.unsqueeze(0)
        embs = self.dropout(self.embed(inputs))
        dec_out, hidden = self.rnn(embs, hidden)
        
        attn_weights = self.attn(dec_out, enc_outs)
        
        context = attn_weights.bmm(enc_outs) 
        concat_input = torch.cat((dec_out, context), -1) 
        concat_output_wo_tanh = self.w(concat_input)
        concat_output = torch.tanh(concat_output_wo_tanh)
        # st = tanh(Wc[ct;st])
        pred = self.prediction(concat_output)

        return pred, hidden,concat_output_wo_tanh

class Seq2seqAttn(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, device):
        super().__init__()        
        self.src_vsz = src_vocab
        self.tgt_vsz = tgt_vocab
        self.embed_dim = 200
        self.hidden_dim = 150
        self.bidirectional = False
        self.dropout = 0.1
        self.n_layers = 1
        self.attn = 'concat'

        self.encoder = EncRNN(self.src_vsz, self.embed_dim, self.hidden_dim, 
                              self.n_layers, self.bidirectional, self.dropout)
        self.decoder = DecRNN(self.tgt_vsz, self.embed_dim, self.hidden_dim, 
                              self.n_layers, self.bidirectional, self.dropout,
                              self.attn)
        self.device = device
        self.use_birnn = self.bidirectional

    def forward(self, srcs, tgts=None, maxlen=150, tf_ratio=0.5,training=True):
        # slen, bsz = srcs.size()
        # tlen = tgts.size(0) if isinstance(tgts, torch.Tensor) else maxlen
        #dec_inputs = torch.ones_like(srcs[1]) * 2 # <eos> is mapped to id=2
        SOS_token = 2
        EOS_token = 1
        batch_size = srcs.size(0)  
        # 解码的长度
        target_len = tgts.size(1)

        enc_outs, hidden = self.encoder(srcs)
        
        dec_inputs = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        dec_inputs = dec_inputs.view(batch_size,1)
        dec_inputs = dec_inputs.to(self.device)
        
        outs = []
        loss = 0
        if self.use_birnn:
            def trans_hidden(hs):
                hs = hs.view(self.n_layers, 2, batch_size, self.hidden_dim)
                hs = torch.stack([torch.cat((h[0], h[1]), 1) for h in hs])
                return hs
            hidden = tuple(trans_hidden(hs) for hs in hidden)
        
        if training:
            for i in range(target_len):
                preds, hidden,_ = self.decoder(dec_inputs, hidden, enc_outs)
                outs.append(preds)
                use_tf = random.random() < tf_ratio
                preds = preds.squeeze(1)
                dec_inputs = tgts[:,i] if use_tf else preds.max(1)[1]
                dec_inputs = dec_inputs.unsqueeze(1)
                
                loss += F.cross_entropy(preds, tgts[:,i], ignore_index=EOS_token)
            return torch.stack(outs), loss
        else:
            beam_size = 5

            terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
            decoder_hidden = hidden
            with open('./decoder_vocab.pickle', 'rb') as file:
                decoder_vocab=pickle.load(file)
            prev_top_sentences.append(Sentence(decoder_hidden,decoder_vocab))
            for i in range(maxlen):
                for sentence in prev_top_sentences:
                    decoder_input = torch.LongTensor([[sentence.last_idx]])
                    decoder_input = decoder_input.to(self.device)

                    decoder_hidden = sentence.decoder_hidden

                    decoder_output, decoder_hidden,_ = self.decoder(decoder_input, decoder_hidden, enc_outs)

                    topv, topi = decoder_output.topk(beam_size)
                    term, top = sentence.addTopk(topi, topv, decoder_hidden, beam_size,decoder_vocab)
                    terminal_sentences.extend(term)
                    next_top_sentences.extend(top)

                next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
                prev_top_sentences = next_top_sentences[:beam_size]
                next_top_sentences = []

            terminal_sentences += [sentence.toWordScore() for sentence in prev_top_sentences]
            terminal_sentences.sort(key=lambda x: x[1], reverse=True)

            n = min(len(terminal_sentences), 15)
            
            return terminal_sentences[:1]


class ADC_Gen(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, device):
        super().__init__()        
        self.src_vsz = src_vocab
        self.tgt_vsz = tgt_vocab
        self.embed_dim = 200
        self.hidden_dim = 150
        self.bidirectional = False
        self.dropout = 0.1
        self.n_layers = 1
        self.attn = 'concat'

        self.encoder = EncRNN(self.src_vsz, self.embed_dim, self.hidden_dim, 
                              self.n_layers, self.bidirectional, self.dropout)
        self.decoder = DecRNN(self.tgt_vsz, self.embed_dim, self.hidden_dim, 
                              self.n_layers, self.bidirectional, self.dropout,
                              self.attn)
        self.device = device
        self.use_birnn = self.bidirectional
        self.charge_pred = nn.Linear(self.hidden_dim*2,62)


    def forward(self, srcs, tgts=None, maxlen=150, tf_ratio=0.5,training=True):
        # slen, bsz = srcs.size()
        # tlen = tgts.size(0) if isinstance(tgts, torch.Tensor) else maxlen
        #dec_inputs = torch.ones_like(srcs[1]) * 2 # <eos> is mapped to id=2
        SOS_token = 2
        EOS_token = 1
        batch_size = srcs.size(0)  
        # 解码的长度
        target_len = tgts.size(1)

        enc_outs, hidden = self.encoder(srcs)
        
        dec_inputs = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        dec_inputs = dec_inputs.view(batch_size,1)
        dec_inputs = dec_inputs.to(self.device)
        
        outs = []
        loss = 0
        if self.use_birnn:
            def trans_hidden(hs):
                hs = hs.view(self.n_layers, 2, batch_size, self.hidden_dim)
                hs = torch.stack([torch.cat((h[0], h[1]), 1) for h in hs])
                return hs
            hidden = tuple(trans_hidden(hs) for hs in hidden)
        gen_out = []
        if training:
            for i in range(target_len):
                preds, hidden,concat_output = self.decoder(dec_inputs, hidden, enc_outs)
                
                gen_out.append(concat_output.squeeze(1))
                outs.append(preds)
                use_tf = random.random() < tf_ratio
                preds = preds.squeeze(1)
                dec_inputs = tgts[:,i] if use_tf else preds.max(1)[1]
                dec_inputs = dec_inputs.unsqueeze(1)
                
                loss += F.cross_entropy(preds, tgts[:,i], ignore_index=EOS_token)
            
            dec_outs = torch.stack(gen_out,1)
            dec_outs = dec_outs.mean(1)
            enc_outs = enc_outs.mean(1)

            charge_hidden = torch.cat([enc_outs,dec_outs],-1)
            charge_out = self.charge_pred(charge_hidden)
            return torch.stack(outs), loss,charge_out
        else:
            beam_size = 5
            terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
            decoder_hidden = hidden
            with open('./decoder_vocab.pickle', 'rb') as file:
                decoder_vocab=pickle.load(file)
            prev_top_sentences.append(Sentence(decoder_hidden,decoder_vocab))
            gen_out = []
            for i in range(maxlen):
                for sentence in prev_top_sentences:
                    decoder_input = torch.LongTensor([[sentence.last_idx]])
                    decoder_input = decoder_input.to(self.device)

                    decoder_hidden = sentence.decoder_hidden

                    decoder_output, decoder_hidden,concat_output = self.decoder(decoder_input, decoder_hidden, enc_outs)
                    gen_out.append(concat_output.squeeze(1))

                    topv, topi = decoder_output.topk(beam_size)
                    term, top = sentence.addTopk(topi, topv, decoder_hidden, beam_size,decoder_vocab)
                    terminal_sentences.extend(term)
                    next_top_sentences.extend(top)

                next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
                prev_top_sentences = next_top_sentences[:beam_size]
                next_top_sentences = []

            terminal_sentences += [sentence.toWordScore() for sentence in prev_top_sentences]
            terminal_sentences.sort(key=lambda x: x[1], reverse=True)

            n = min(len(terminal_sentences), 15)

            dec_outs = torch.stack(gen_out,1)
            dec_outs = dec_outs.mean(1)
            enc_outs = enc_outs.mean(1)

            charge_hidden = torch.cat([enc_outs,dec_outs],-1)
            charge_out = self.charge_pred(charge_hidden)

            return terminal_sentences[:1],charge_out



class Gen(nn.Module):
    def __init__(self,device,src_vocab, tgt_vocab,use_birnn=True):
        super(Gen, self).__init__()
        self.embed_dim = 200
        self.hidden_dim = 150    
        self.device = device
        self.embed = nn.Embedding(src_vocab, self.embed_dim)
        self.embed.weight.requires_grad = True

        self.sc_gen = Seq2seqAttn(src_vocab, tgt_vocab, self.device)
        self.adc_gen = ADC_Gen(src_vocab, tgt_vocab, self.device)

    def forward(self, sc_source,adc_source,sc_target,adc_target,training=True):
        # 默认batch为1
        if training:
            out1,loss_sc = self.sc_gen(sc_source,sc_target)
            out2,loss_adc,charge_out = self.adc_gen(adc_source,adc_target)
            return loss_sc,loss_adc,charge_out
        else:
            out1 = self.sc_gen(sc_source,sc_target,training = training)
            out2,charge_out = self.adc_gen(adc_source,adc_target,training = training)
            return out1,out2,charge_out

            
        



