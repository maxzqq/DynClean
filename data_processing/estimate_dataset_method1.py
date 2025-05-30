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
label_num_dic = {
    'conll03': 4,
    'twitter': 10,
    'bc5cdr': 2,
    'wiki': 4,
    'webpage': 4,
    'onto': 18,
    'onto2': 18,
}
num_ner_labels = label_num_dic[dataset_name]

print("Procssing dataset: ", dataset_name)

with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/train-ds-samples.json', 'r') as f:
    samples = json.load(f)
neg_guids = []
pos_guids = {}
for sample in tqdm(samples):
    span_labels = sample['spans_label']
    guids = sample['guids']
    for guid, span_label in zip(guids, span_labels):
        if span_label == 0:
            neg_guids.append(guid)
        else:
            if span_label not in pos_guids:
                pos_guids[span_label] = []
            pos_guids[span_label].append(guid)
# obtain guids for fake class
total_pos_num = sum([len(pos_guids[k]) for k in pos_guids])
sample_pos_num = sum([len(pos_guids[k]) for k in pos_guids]) / len(pos_guids)
# nc_neg_guids = random.sample(neg_guids, int(sample_pos_num))
nc_neg_guids = random.sample(neg_guids, int(len(neg_guids)*0.01))
nc_pos_guids = []
for key, value in pos_guids.items():
    num = int(len(value)/total_pos_num * sample_pos_num) # sample number for each label by label distribution
    nc_pos_guids.extend(random.sample(value, num)) # guids for each label

# all together
aum_guids = nc_neg_guids + nc_pos_guids
def create_aum_data(aum_guids, samples, num_ner_labels):
    # change the class label for the fake class
    new_class = num_ner_labels + 1
    for i in trange(len(samples)):
        sample = samples[i]
        guids = sample['guids']
        span_labels = sample['spans_label']
        for j in range(len(guids)):
            guid = guids[j]
            if guid in aum_guids:
                span_labels[j] = new_class
        samples[i]['spans_label'] = span_labels
    return samples
samples = create_aum_data(aum_guids, samples, num_ner_labels)

# save the data
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/aum/train-ds-samples-aum-new.json', 'w') as f:
    json.dump(samples, f)
# save the guids
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/aum/fake_class_guids-new.pkl', 'wb') as f:
    pickle.dump(aum_guids, f)
print("----- Saving aum data sucessfully!")