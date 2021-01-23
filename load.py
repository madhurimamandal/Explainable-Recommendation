import torch
import re
import os
import pickle
import unicodedata
import numpy as np
import scipy.sparse as sp

from config import MAX_LENGTH, save_dir

PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3

class Voc:
    def __init__(self, name):
        self.name = name
        
        with open(os.path.join(save_dir, 'word_vocab.pkl'), 'rb') as fp:
            self.word2idx, self.idx2word = pickle.load(fp)
        self.n_words = len(self.word2idx)

def tokenize(path, corpus_name, voc):
    print("Reading {}".format(path))
    
    with open(os.path.join(save_dir, 'user_item.pkl'), 'rb') as fp:
        user_dict, item_dict = pickle.load(fp)
    pairs = []

    users=[]
    items=[]
    user_items=[]
    with open(path, 'r') as f:
        for l in f.readlines():
            l = eval(l)
            user = l['reviewerID']
            item = l['asin']
            user_id = user_dict[user]
            item_id = item_dict[item]

            review = l['reviewText_tok']
            sent = ['<str>'] + review + ['<eos>']
            summary = l['summary_tok']
            title = []
            for w in l['title_tok']:
                if w in voc.word2idx:
                    title.append(w)
            if len(summary) <= 0 or len(title) <= 0:
                continue
            aux = [user_id, item_id, summary, title]
            pair = [aux, sent]
            pairs.append(pair)
            users.append(user_id)
            items.append(item_id)
            user_items.append([user_id, item_id])

    train_mat = sp.dok_matrix((np.max(users)+1, np.max(items)+1), dtype=np.float32)

    for x in user_items:
	    train_mat[x[0], x[1]] = 1.0

    return pairs, train_mat

def prepareData(corpus_name):
    voc = Voc(corpus_name)
    train_pairs, train_mat = tokenize(os.path.join(save_dir, 'train_tok.json'), corpus_name, voc)
    valid_pairs, valid_mat = tokenize(os.path.join(save_dir, 'valid_tok.json'), corpus_name, voc)
    test_pairs, test_mat = tokenize(os.path.join(save_dir, 'test_tok.json'), corpus_name, voc)

    torch.save(train_pairs, os.path.join(save_dir, '{!s}.tar'.format('train_pairs')))
    torch.save(valid_pairs, os.path.join(save_dir, '{!s}.tar'.format('valid_pairs')))
    torch.save(test_pairs, os.path.join(save_dir, '{!s}.tar'.format('test_pairs')))
    return voc, train_pairs, valid_pairs, test_pairs, train_mat


def loadPrepareData(corpus_name):
    try:
        print("Start loading training data ...")
        voc = Voc(corpus_name)
        train_pairs = torch.load(os.path.join(save_dir, 'train_pairs.tar'))
        valid_pairs = torch.load(os.path.join(save_dir, 'valid_pairs.tar'))
        test_pairs = torch.load(os.path.join(save_dir, 'test_pairs.tar'))
        train_mat = torch.load(os.path.join(save_dir, 'train_mat.tar'))
        
    except FileNotFoundError:
        print("Saved data not found, start preparing training data ...")
        voc, train_pairs, valid_pairs, test_pairs, train_mat = prepareData(corpus_name)
        torch.save(train_mat, os.path.join(save_dir, "",'train_mat.tar'))
    return voc, train_pairs, valid_pairs, test_pairs, train_mat

if __name__ == '__main__':
    corpus_name = 'Electronics'
    loadPrepareData(corpus_name)
    
