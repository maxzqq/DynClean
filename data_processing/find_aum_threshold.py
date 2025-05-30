import json
import os
from tqdm import tqdm,trange
import numpy as np
import pandas as pd
import pickle
import torch
import pickle
import matplotlib.pyplot as plt
import random

# CoNLL 03
# load incomplete and inaccurate guids
dataset = 'conll03'
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/incomplete.pkl', 'rb') as f:
    incomplete_guids = pickle.load(f)
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/inaccurate.pkl', 'rb') as f:
    inaccurate_guids = pickle.load(f)
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/aum/fake_class_guids.pkl', 'rb') as f:
    fake_guids = pickle.load(f)
# 分离出ds数据中的entity span guids和非entity span guids
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/train-ds-samples.json', 'r') as f:
    ds_data = json.load(f)
entity_guids = []
non_entity_guids = []
for sample in ds_data:
    guids = sample['guids']
    spans_label = sample['spans_label']
    for i in range(len(spans_label)):
        if spans_label[i] != 0:
            entity_guids.append(guids[i])
        else:
            non_entity_guids.append(guids[i])
print(len(entity_guids), len(non_entity_guids))
print("----- Loading entity guids and non_entity_guids sucessfully!")
print("False Negatives: ", len(incomplete_guids))
print("False Positives: ", len(inaccurate_guids))