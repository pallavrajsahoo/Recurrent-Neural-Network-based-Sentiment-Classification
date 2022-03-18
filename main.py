"""
Sentiment analysis with RNN and BERT embedding
Ting-Yao Hu, 2021/03
"""

import os.path as osp
import argparse
from utils import load_pretrained_bert, bert_emb_sentence, accuracy
import h5py
import numpy as np
import tqdm
import time

import torch
from torch import optim, nn

idx2label = ["positive", "neutral", "negative"]
label2idx = {label: idx for idx, label in enumerate(idx2label)}

BERT_EMB_SIZE = 768
OUT_HELDOUT_PATH = "heldout_pred.txt"

""" Adapted from homework7, spring 2020 """
class Classifier(nn.Module):
    def __init__(self, rnn_in_dim, rnn_hid_dim):
        super(Classifier, self).__init__()

        self.rnn_in_dim = rnn_in_dim
        self.rnn_hid_dim = rnn_hid_dim

        # Layers
        self.rnn = nn.RNN(rnn_in_dim, rnn_hid_dim)
        self.rnn2logit = nn.Linear(rnn_hid_dim, 3)
        #self.rnn2logit = nn.Linear(rnn_hid_dim, 1)

    def init_rnn_hid(self):
        """Initial hidden state."""
        return torch.zeros(1, 1, self.rnn_hid_dim)

    def forward(self, feat_seq):
        """Feeds the words into the neural network and returns the value
        of the output layer."""
        rnn_outs, _ = self.rnn(feat_seq.unsqueeze(1), self.init_rnn_hid())
                                      # (seq_len, 1, rnn_hid_dim)
        logit = self.rnn2logit(rnn_outs[-1]) # (1 x 3)
        return logit


###############
##   Tasks   ##
###############

###
## - text_fn: input text file name
## - out_fn: h5 file name
###

def save_bert_embedding(text_fn, out_fn):
    ### data
    texts = [l.strip() for l in open(text_fn, 'r')]

    ### save bert embedding to h5 file
    h5_obj = h5py.File(out_fn,'w')
    dt = h5py.special_dtype(vlen=np.dtype('float32'))
    dataset = h5_obj.create_dataset('embedding',(len(texts),), dtype=dt)
    dataset2 = h5_obj.create_dataset('token_num',(len(texts),), dtype='int')
    model, tokenizer = load_pretrained_bert()
    pbar = tqdm.tqdm(texts)
    for i, text in enumerate(pbar):
        emb, tokens = bert_emb_sentence(text, model, tokenizer)
        save_bert_to_h5(h5_obj, i, emb)
        continue

    h5_obj.close()


###
## TODO: Task1
##  - h5_obj: 'h5 file object'
##  - idx: 'int', sentence index (the idx-th sentence)
##  - emb: 'torch.FloatTensor', (1 x token_number x bert_embedding_size), bert embedding of the idx-th sentence
###
def save_bert_to_h5(h5_obj, idx, emb):
    h5_obj['embedding'][idx] = torch.flatten(emb)
    pass

###
## TODO: Task2
##  - h5_obj: 'h5 file object'
##  - idx: 'int', sentence index (the idx-th sentence)
## output:
##  - feat: 'torch.FloatTensor', (token_number x bert_embedding_size)
###
def load_bert_from_h5(h5_obj, idx):
    feat = torch.tensor(h5_obj['embedding'][idx].reshape(-1, 768))
    return feat

###
## TODO: Task3
## - logit: 'torch.FloatTensor'
## output:
## - pred: 'int'
###
def pred_from_logit(logit):
    '''
    if(logit[0].item() > 0):
        return 0
    elif(logit[0].item() == 0):
        return 1
    elif(logit[0].item() < 0):
        return 2
    '''

    pred = torch.argmax(logit).item()
    return pred



if __name__=='__main__':

    ###############
    ## arguments ##
    ###############
    parser = argparse.ArgumentParser()
    ### args for data
    parser.add_argument('-train_fn', default='data/dev_text.txt', type=str)
    parser.add_argument('-train_lab_fn', default='data/dev_label.txt', type=str)
    parser.add_argument('-test_fn', default='data/heldout_text.txt', type=str)
    parser.add_argument('-train_h5_fn', default='data/bert.h5', type=str)
    parser.add_argument('-test_h5_fn', default='data/bert_test.h5', type=str)

    ### args for classifier training
    parser.add_argument("-rnn_hid_dim", default=26, type=int,
                        help="Dimentionality of RNN hidden state")
    parser.add_argument("-epochs", default=4, type=int,
                        help="Number of epochs")
    args = parser.parse_args()


    #####################################################
    ## extracting bert embeddings, saving as .h5 format ##
    #####################################################

    if not osp.exists(args.train_h5_fn):
        save_bert_embedding(args.train_fn, args.train_h5_fn)
    if not osp.exists(args.test_h5_fn):
        save_bert_embedding(args.test_fn, args.test_h5_fn)


    ########################
    ## pytorch classifier ##
    ########################
    clf = Classifier(BERT_EMB_SIZE, args.rnn_hid_dim)
    optimizer = optim.Adam(clf.parameters())
    #ce_loss = nn.CrossEntropyLoss()
    #ce_loss = nn.MSELoss()
    ce_loss = nn.L1Loss()

    ################
    ##  training  ##
    ################
    labs = [label2idx[l.strip()] for l in open(args.train_lab_fn,'r')]
    h5_obj = h5py.File(args.train_h5_fn, 'r')
    clf.train()
    for epoch in range(args.epochs):

        print('Epoch:', epoch+1)
        for idx in range(len(labs)):
            lab = labs[idx]
            feat = load_bert_from_h5(h5_obj, idx)

            logit = clf.forward(feat)
            loss = ce_loss(logit, torch.LongTensor([lab]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    h5_obj.close()


    ###############
    ### testing ###
    ###############
    pred_list = []
    h5_obj = h5py.File(args.test_h5_fn, 'r')
    clf.eval()
    test_num = len(h5_obj['embedding'])
    for i in range(test_num):
        feat = load_bert_from_h5(h5_obj, i)

        logit = clf.forward(feat)
        pred = pred_from_logit(logit)
        pred_list.append(idx2label[pred])
    h5_obj.close()

    out = open(OUT_HELDOUT_PATH, 'w')
    for pred in pred_list: out.write(pred+'\n')
    out.close()
