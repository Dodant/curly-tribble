import csv

import torch

from construct_ds import *
from pretrain_embed import *


def make_submission(reviewer, path, test_path='dataset/test'):
    vocab = make_vocab()

    csvf = open(f'submission/{path}.csv', 'w')
    writer = csv.writer(csvf)
    writer.writerow(['id', 'sentiment'])
    for filename in range(2000):
        if filename == '.ipynb_checkpoints': continue
        with open(os.path.join(test_path, str(filename)), 'r') as f:
            text = f.read()
            clean_text = cleaning(text).split()
            text_seq = embed_seq(clean_text, vocab)
            text_seq = torch.IntTensor(text_seq).cuda()
            text_seq = torch.unsqueeze(text_seq, 0)
            y = reviewer(text_seq)
            y = 1 if y.item() >= 0.5 else 0
            writer.writerow([f'{filename}', f'{y}'])
    csvf.close()


def embed_make_submission(reviewer, path, dict_dict, test_path='dataset/test'):
    csvf = open(f'submission/{path}.csv', 'w')
    writer = csv.writer(csvf)
    writer.writerow(['id', 'sentiment'])
    for filename in range(2000):
        if filename == '.ipynb_checkpoints': continue
        with open(os.path.join(test_path, str(filename)), 'r') as f:
            text = f.read()
            clean_text = cleaning(text).split()
            text_seq = []
            if len(clean_text) < 100:
                temp = make_pretrained_seq(clean_text, dict_dict)
                temp = temp + [dict_dict['</s>'] for i in range(100 - len(temp))]
                text_seq.append(temp)
            else:
                temp = make_pretrained_seq(clean_text, dict_dict)
                text_seq.append(temp[:100])
            text_seq = torch.Tensor(text_seq[0]).cuda()
            text_seq = torch.unsqueeze(text_seq, 0)
            y = reviewer(text_seq)
            y = 1 if y.item() >= 0.5 else 0
            writer.writerow([f'{filename}', f'{y}'])
    csvf.close()