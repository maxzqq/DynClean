import json
import os
from tqdm import tqdm,trange
import numpy as np
import pandas as pd
import pickle
import pickle


# processing conll format txt data file
file_path = '/home/qi/Projects/NER/ds_ner/data/twitter/train.txt'
tokens_list = []
ner_tags_list = []
with open(file_path, 'r') as f:
    tokens = []
    ner_tags = []
    for line in f:
        if line == '\n':
            tokens_list.append(tokens[:-1])
            ner_tags_list.append(ner_tags[:-1])
            tokens = []
            ner_tags = []
        else:
            token, ner_tag = line.strip().split()
            tokens.append(token)
            ner_tags.append(ner_tag)

# load ds data
with open('/home/qi/Projects/NER/ds_ner/data/twitter/train-ds.json', 'r') as f:
    ds_data = json.load(f)
gs_data = []
for sample in ds_data:
    new_sample = {}
    tokens = sample['tokens']
    if tokens in tokens_list:
        new_sample['tokens'] = tokens
        idx = tokens_list.index(tokens)
        assert tokens == tokens_list[idx]
        labels = ner_tags_list[idx]
        assert len(labels) == len(tokens)
        new_sample['token_labels'] = labels
        gs_data.append(new_sample)
print(len(gs_data))
print(len(ds_data))




