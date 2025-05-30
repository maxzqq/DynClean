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

# Hyperparameters
dataset = 'conll03'
aum_num = 2
NEG_THES = 0.1
POS_THES = 0

print("Procssing dataset: ", dataset)
print("NEG_THES: ", NEG_THES)
print("POS_THES: ", POS_THES)

# ===== load noisy guids =====
# load incomplete and inaccurate guids
dataset = 'conll03'
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/incomplete.pkl', 'rb') as f:
    incomplete_guids = pickle.load(f)
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/inaccurate.pkl', 'rb') as f:
    inaccurate_guids = pickle.load(f)
# with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/aum/fake_class_guids-new-other-label.pkl', 'rb') as f:
#     fake_guids = pickle.load(f)
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
print("----- Loading entity guids and non_entity_guids sucessfully!")
print("Positive #:", len(entity_guids))
print("Negative #:", len(non_entity_guids))
# print(len(entity_guids), len(non_entity_guids))
print("----- Loading noisy guids sucessfully!")
print("False Negatives: ", len(incomplete_guids))
print("False Positives: ", len(inaccurate_guids))

# ===== Get the noisy negative guids =====
# 1. Threshold AUM values
# fake data 用于estimate threshold的aum数据
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/aum/fake_class_guids-new-other-label.pkl', 'rb') as f:
    aum_guids = pickle.load(f)
aum = pd.read_csv(f'/home/qi/Projects/NER/ds_ner/out/new_{dataset}_bert_td_new/dy_log/aum_{aum_num}/aum_values.csv')
neg_aum = aum[aum['sample_id'].isin(non_entity_guids)]
fake_aum = neg_aum[neg_aum['sample_id'].isin(aum_guids)]
fake_aum = fake_aum.sort_values(by=['aum'], ascending=False)
# fake_aum[:int(len(fake_aum)*0.5)]
thres = fake_aum.iloc[int(len(fake_aum)*NEG_THES)]['aum']
print("Negattive AUM Threshold value: ", thres)

# 2. Filter out the noisy negative guids by threshold AUM values
# 真实的AUM
aum = pd.read_csv(f'/home/qi/Projects/NER/ds_ner/out/new_{dataset}_bert/dy_log/aum_{aum_num}/aum_values.csv')
hard_neg_aum = aum[aum['sample_id'].isin(non_entity_guids)]
hard_neg_aum = hard_neg_aum.sort_values(by=['aum'], ascending=False)
# iterative all the rows in hard_neg_aum
neg_rm_guids = []
for i in range(len(hard_neg_aum)):
    row = hard_neg_aum.iloc[i]
    guid = row['sample_id']
    aum_value = row['aum']
    if aum_value < thres:
        neg_rm_guids.append(guid)
print("Removed total neg #:", len(neg_rm_guids))
inter_sec = list(set(neg_rm_guids).intersection(incomplete_guids))
print("Removed false neg #:", len(inter_sec))

# ===== Get the noisy positive guids =====
# 1. Threshold AUM values
# fake data 用于estimate threshold的aum数据
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/aum/fake_class_guids-new-other-label.pkl', 'rb') as f:
    aum_guids = pickle.load(f)
aum = pd.read_csv(f'/home/qi/Projects/NER/ds_ner/out/new_{dataset}_bert_td_new/dy_log/aum_{aum_num}/aum_values.csv')
pos_aum = aum[aum['sample_id'].isin(entity_guids)]
fake_aum = pos_aum[pos_aum['sample_id'].isin(aum_guids)]
fake_aum = fake_aum.sort_values(by=['aum'], ascending=False)
thre_aum = fake_aum.iloc[int(len(fake_aum)*POS_THES)]['aum']
print(thre_aum)
# 真实的AUM
aum = pd.read_csv(f'/home/qi/Projects/NER/ds_ner/out/new_{dataset}_bert/dy_log/aum_{aum_num}/aum_values.csv')
hard_pos_aum = aum[aum['sample_id'].isin(entity_guids)]
hard_pos_aum = hard_pos_aum.sort_values(by=['aum'], ascending=False)

pos_rm_guids = []
for i in range(len(hard_pos_aum)):
    row = hard_pos_aum.iloc[i]
    guid = row['sample_id']
    aum_value = row['aum']
    if aum_value < thre_aum:
        pos_rm_guids.append(guid)
print("Removed total pos #:", len(pos_rm_guids))
inter_sec = list(set(pos_rm_guids).intersection(inaccurate_guids))
print("Removed false pos #:", len(inter_sec))

# ====== Filter out both noisy negative and positive guids ======
all_rm_guids = list(set(neg_rm_guids + pos_rm_guids))
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/train-ds-samples.json', 'r') as f:
    ds_data = json.load(f)
new_ds = []
for i in trange(len(ds_data)):
    sample = ds_data[i]
    guids = sample['guids']
    spans_label = sample['spans_label']
    spans = sample['spans']
    new_guids = []
    new_spans_label = []
    nwe_spans = []
    for j in range(len(guids)):
        if guids[j] not in all_rm_guids:
            # 不在indices中的guids才是我们要的
            new_guids.append(guids[j])
            new_spans_label.append(spans_label[j])
            nwe_spans.append(spans[j])
    ds_data[i]['guids'] = new_guids
    ds_data[i]['spans_label'] = new_spans_label
    ds_data[i]['spans'] = nwe_spans
    if len(ds_data[i]['spans']) > 0:
        new_ds.append(ds_data[i])
print(f"New ds data length: {len(new_ds)}")
with open(f"/home/qi/Projects/NER/ds_ner/data/{dataset}/train-ds-samples-removed-noise-bert.json", "w") as f:
        json.dump(new_ds, f)