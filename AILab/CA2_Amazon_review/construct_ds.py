import os
import string
from collections import Counter

import nltk
import numpy as np
import torch

nltk.download('wordnet')


# 어근 check
def cleaning(text):
    wlem = nltk.WordNetLemmatizer()

    punc = set(string.punctuation)
    punc.remove('!')
    punc.remove('?')
    punc.update([i for i in range(10)])
    text = text.lower()
    words = text.split()

    clean_text = []

    for i in words:
        t = ''.join([x for x in i.encode('ascii', 'ignore').decode('ascii') if x not in punc])
        if t == 'nt': t = 'not'
        if t == 've': t = 'have'
        if t == 're': t = 'are'
        if t in ['s', 'a', 'an', 'the'] or t.isnumeric() or len(t) == 0: continue
        if t in ['as', 'has', 'was']:
            clean_text.append(t)
            continue
        if t == 'does':
            clean_text.append(t)
            continue
        lemma = wlem.lemmatize(t)
        clean_text.append(lemma)
    return " ".join([i for i in clean_text])


def make_vocab(positive_path='dataset/train/positive', negative_path='dataset/train/negative', length=10000):
    temp_bag = []

    for filename in os.listdir(positive_path):
        if filename == '.ipynb_checkpoints': continue
        with open(os.path.join(positive_path, filename), 'r') as f:
            text = f.read()
            clean_text = cleaning(text).split()
            temp_bag.extend(clean_text)
        with open(os.path.join(negative_path, filename), 'r') as f:
            text = f.read()
            clean_text = cleaning(text).split()
            temp_bag.extend(clean_text)

    vocab = {word: i + 2 for i, word in enumerate(list(dict(Counter(temp_bag).most_common(length-2)).keys()))}
    vocab['<unk>'], vocab['<pad>'] = 0, 1
    return vocab


def trim_seq(sequence, vocab, max_length=100):
    sequence_bag = []
    if len(sequence) < max_length:
        sequence_bag.append(embed_seq(sequence, vocab, max_length=max_length))
    else:
        g = np.reshape(sequence[:-(len(sequence) % max_length)], (-1, max_length))
        for text in g:
            sequence_bag.append(embed_seq(text, vocab, max_length=max_length))
    return sequence_bag


def embed_seq(sequence, vocab, max_length=100):
    embed_seq_ls = []
    for i in range(max_length):
        if i >= len(sequence):
            embed_seq_ls.append(1)
        else:
            if sequence[i] in vocab:
                embed_seq_ls.append(vocab[sequence[i]])
            else:
                embed_seq_ls.append(0)
    return embed_seq_ls
