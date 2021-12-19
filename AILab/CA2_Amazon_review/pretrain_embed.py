import os
import numpy as np
import torch

from construct_ds import *



def make_pretrained_seq(text_seq, dict_dict):
    embedding_seq = []
    dict_keys = list(dict_dict.keys())
    for x in text_seq:
        if x in dict_keys:
            embedding_seq.append(dict_dict[x])
        else:
            embedding_seq.append(dict_dict['</s>'])
    return embedding_seq


def embed_trim_seq(sequence, dict_dict, max_length=100):
    sequence_bag = []
    if len(sequence) < max_length:
        temp = make_pretrained_seq(sequence, dict_dict)
        temp = temp + [dict_dict['</s>'] for i in range(max_length - len(temp))]
        sequence_bag.append(temp)
    else:
        g = np.reshape(sequence[:-(len(sequence) % max_length)], (-1, max_length))
        for text in g:
            sequence_bag.append(make_pretrained_seq(text, dict_dict))
    return sequence_bag