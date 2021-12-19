import os

from torchtext.legacy import data
from construct_ds import *

positive_list = []
positive_path='dataset/train/positive'

for filename in os.listdir(positive_path):
    if filename == '.ipynb_checkpoints': continue
    with open(os.path.join(positive_path, filename), 'r') as f:
        text = f.read()
        clean_text = cleaning(text)
        clean_text = clean_text.split()

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  lower=True,
                  batch_first=True,
                  fix_length=20)

TEXT.build_vocab(positive_list, min_freq=5, max_size=10000)

print(TEXT.vocab.stoi)