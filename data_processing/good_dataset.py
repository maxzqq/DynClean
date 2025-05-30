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

dataset_name = 'conll03'
# 
# load incomplete and inaccurate guids
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/incomplete.pkl', 'rb') as f:
    incomplete_guids = pickle.load(f)
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/inaccurate.pkl', 'rb') as f:
    inaccurate_guids = pickle.load(f)

##### load dataset
with open('/home/qi/Projects/NER/ds_ner/data/conll03/train-ds-samples-removed-noise.json', 'r') as f:
    ds_data = json.load(f)

# get all guids in ds_dataset
ds_guids = []
for sample in ds_data:
    ds_guids.extend(sample['guids'])

# intersection of ds_guids and inaccurate
print("===> For inaccurate")
inaccu_guids = list(set(ds_guids).intersection(set(inaccurate_guids)))
print(len(inaccu_guids), len(inaccurate_guids))
# intersection of ds_guids and incomplete
print("===> For incomplete")
incom_guids = list(set(ds_guids).intersection(set(incomplete_guids)))
print(len(incom_guids), len(incomplete_guids))
