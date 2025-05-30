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

dataset_name = 'wiki'
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
NEG_NUM_TYPE = False # 'percent' or 'number'

# generate label_id list
# label_ids = [i for i in range(num_ner_labels+1)]

print("Procssing dataset: ", dataset_name)
print("Class number: ", num_ner_labels)

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
# sampling
# 1. 假的负样本为1%的负样本
if NEG_NUM_TYPE:
    print("Negative sample number type: percent")
    nc_neg_guids = random.sample(neg_guids, int(len(neg_guids)*0.01))
else:
    # 2. 假的负样本和正样本一样多
    nc_neg_guids = random.sample(neg_guids, int(sample_pos_num))
    nc_pos_guids = {} # new class pos guids dict
nc_pos_guids_list = [] # new class pos guids list
for key, value in pos_guids.items():
    num = int(len(value)/total_pos_num * sample_pos_num) # sample number for each label by label distribution
    nc_pos_guids_list.extend(random.sample(value, num)) # guids for each label
    # nc_pos_guids[key] = random.sample(value, num) # fake data guids for each label
    # nc_pos_guids_list.extend(nc_pos_guids[key])

# get guids->label_id dictionary
aum_guids = nc_neg_guids + nc_pos_guids_list
# nc_guids_label_id = {}
# for guid in nc_neg_guids:
#     nc_guids_label_id[guid] = 0
# for key, value in nc_pos_guids.items():
#     for guid in value:
#         nc_guids_label_id[guid] = key

# ====== old method, directly assign new class (class number + 1) to the fake class ======
# def create_aum_data(aum_guids, samples, num_ner_labels):
#     # change the class label for the fake class
#     new_class = num_ner_labels + 1
#     for i in trange(len(samples)):
#         sample = samples[i]
#         guids = sample['guids']
#         span_labels = sample['spans_label']
#         for j in range(len(guids)):
#             guid = guids[j]
#             if guid in aum_guids:
#                 span_labels[j] = new_class
#         samples[i]['spans_label'] = span_labels
#     return samples
# samples = create_aum_data(aum_guids, samples, num_ner_labels)

# ====== new method =======
# 1. Positive, assign new class that is not the original label_id
# 2. Negative, assign new class that is not the original label_id, i.e., one of the positive label_id

new_class_id = num_ner_labels + 1

for i in trange(len(samples)):
    # get one sentence alll span samples
    sample = samples[i]
    guids = sample['guids']
    span_labels = sample['spans_label']
    for j in range(len(guids)):
        guid = guids[j]
        if guid in aum_guids:
            # get the original label_id
            original_label = span_labels[j]
            if original_label == 0:
                # negative, assign a psotive class
                new_label_id = random.randint(1, num_ner_labels)
            else:
                # positive, assign new class
                new_label_id = new_class_id
            assert new_label_id != original_label # make sure the new label_id is not the original one
            # update the span_labels
            span_labels[j] = new_label_id
    # update the sample's span_labels
    samples[i]['spans_label'] = span_labels

# save the data
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/aum/train-ds-samples-aum-new-method2.json', 'w') as f:
    json.dump(samples, f)
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/aum/fake_class_guids-new-method2.pkl', 'wb') as f:
    pickle.dump(aum_guids, f)
print("----- Saving aum data sucessfully!")