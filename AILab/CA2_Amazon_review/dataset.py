import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from construct_ds import *
from pretrain_embed import *



class ReviewDataset(Dataset):
    def __init__(self, vocab,
                 positive_path='dataset/train/positive',
                 negative_path='dataset/train/negative',
                 word_limit=100):
        self.positive_list = []
        self.negative_list = []

        for filename in os.listdir(positive_path):
            if filename == '.ipynb_checkpoints': continue
            with open(os.path.join(positive_path, filename), 'r') as f:
                text = f.read()
                clean_text = cleaning(text).split()
                for seq_block in trim_seq(clean_text, vocab, max_length=word_limit):
                    self.positive_list.append(torch.IntTensor(seq_block))
            with open(os.path.join(negative_path, filename), 'r') as f:
                text = f.read()
                clean_text = cleaning(text).split()
                for seq_block in trim_seq(clean_text, vocab, max_length=word_limit):
                    self.negative_list.append(torch.IntTensor(seq_block))

    def __len__(self):
        return len(self.positive_list) + len(self.negative_list)

    def __getitem__(self, idx):
        if idx < len(self.positive_list)-1:
            return self.positive_list[idx], torch.from_numpy(np.array([1])).float()
        else:
            return self.negative_list[idx-len(self.positive_list)], torch.from_numpy(np.array([0])).float()


class Embed_ReviewDataset(Dataset):
    def __init__(self,
                 dict_dict,
                 positive_path='dataset/train/positive',
                 negative_path='dataset/train/negative'):
        self.positive_list = []
        self.negative_list = []
        for filename in os.listdir(positive_path):
            if filename == '.ipynb_checkpoints':
                continue
            with open(os.path.join(positive_path, filename), 'r') as f:
                text = f.read()
                clean_text = cleaning(text).split()
                for seq_block in embed_trim_seq(clean_text, dict_dict):
                    self.positive_list.append(torch.Tensor(seq_block))
            with open(os.path.join(negative_path, filename), 'r') as f:
                text = f.read()
                clean_text = cleaning(text).split()
                for seq_block in embed_trim_seq(clean_text, dict_dict):
                    self.negative_list.append(torch.Tensor(seq_block))

    def __len__(self):
        return len(self.positive_list) + len(self.negative_list)

    def __getitem__(self, idx):
        if idx < len(self.positive_list)-1:
            return self.positive_list[idx], torch.from_numpy(np.array([1])).float()
        else:
            return self.negative_list[idx-len(self.positive_list)], torch.from_numpy(np.array([0])).float()