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

dataset_name = 'bc5cdr'

# load incomplete and inaccurate guids
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/incomplete.pkl', 'rb') as f:
    incomplete_guids = pickle.load(f)
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/inaccurate.pkl', 'rb') as f:
    inaccurate_guids = pickle.load(f)
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/aum/fake_class_guids.pkl', 'rb') as f:
    fake_guids = pickle.load(f)
# 分离出ds数据中的entity span guids和非entity span guids
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/train-ds-samples.json', 'r') as f:
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

df = pd.read_json(f'/home/qi/Projects/NER/ds_ner/out/{dataset_name}/dy_log/td_df.jsonl', lines=True)
# 1. 先处理Negatives
# 获取hard negatives
print("===> For Negatives")
neg_df = df[df['guid'].isin(non_entity_guids)]
neg_df = neg_df.sort_values(by=['confidence'], ascending=True)
selected_df = neg_df.head(n=int(0.3 * len(neg_df)))
hard_neg = list(selected_df['guid'])

# 获取对应的fake aum threshold
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/aum/fake_class_guids.pkl', 'rb') as f:
    aum_guids = pickle.load(f)
aum = pd.read_csv(f'/home/qi/Projects/NER/ds_ner/out/{dataset_name}_aum/dy_log/aum_values.csv')
neg_aum = aum[aum['sample_id'].isin(non_entity_guids)]
fake_aum = neg_aum[neg_aum['sample_id'].isin(aum_guids)]
fake_aum = fake_aum.sort_values(by=['aum'], ascending=False)
thre_aum = fake_aum.iloc[int(len(fake_aum)*0.1)]['aum']

# 获得需要被删掉的hard negatives的guids
# 真实的AUM
aum = pd.read_csv(f'/home/qi/Projects/NER/ds_ner/out/{dataset_name}/dy_log/aum_values.csv')
hard_neg_aum = aum[aum['sample_id'].isin(hard_neg)]
hard_neg_aum = hard_neg_aum.sort_values(by=['aum'], ascending=False)
# iterative all the rows in hard_neg_aum
rm_hard_neg_guids = []
for i in range(len(hard_neg_aum)):
    row = hard_neg_aum.iloc[i]
    guid = row['sample_id']
    aum_value = row['aum']
    if aum_value < thre_aum:
        rm_hard_neg_guids.append(guid)
print("删掉的总的hard negatives数目:",len(rm_hard_neg_guids))
inter_sec = list(set(rm_hard_neg_guids).intersection(incomplete_guids))
print("删掉的hard negatives中error的数目:", len(inter_sec))

# 2. 再处理Positives
# 获取hard positives
print("===> For Positives")
pos_df = df[df['guid'].isin(entity_guids)]
pos_df = pos_df.sort_values(by=['confidence'], ascending=True)
selected_df = pos_df.head(n=int(0.3 * len(pos_df)))
hard_pos = list(selected_df['guid'])
hard_rm = list(set(hard_pos).intersection(inaccurate_guids))
print(f"========0.3 hard to learn Positives=========")
print("Total False Positives #: ", len(inaccurate_guids))
print(len(hard_rm), len(hard_pos))

# 获取对应的fake aum threshold
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/aum/fake_class_guids.pkl', 'rb') as f:
    aum_guids = pickle.load(f)
aum = pd.read_csv(f'/home/qi/Projects/NER/ds_ner/out/{dataset_name}_aum/dy_log/aum_values.csv')
pos_aum = aum[aum['sample_id'].isin(entity_guids)]
fake_aum = pos_aum[pos_aum['sample_id'].isin(aum_guids)]
fake_aum = fake_aum.sort_values(by=['aum'], ascending=False)
thre_aum = fake_aum.iloc[int(len(fake_aum)*0.2)]['aum']

# 获得需要被删掉的hard positives的guids
# 真实的AUM
aum = pd.read_csv(f'/home/qi/Projects/NER/ds_ner/out/{dataset_name}/dy_log/aum_values.csv')
hard_neg_aum = aum[aum['sample_id'].isin(hard_pos)]
hard_neg_aum = hard_neg_aum.sort_values(by=['aum'], ascending=False)
# iterative all the rows in hard_neg_aum
rm_hard_pos_guids = []
for i in range(len(hard_neg_aum)):
    row = hard_neg_aum.iloc[i]
    guid = row['sample_id']
    aum_value = row['aum']
    if aum_value < thre_aum:
        rm_hard_pos_guids.append(guid)
print(len(rm_hard_pos_guids), len(hard_neg_aum))
inter_sec = list(set(rm_hard_pos_guids).intersection(inaccurate_guids))
print("删掉的hard positives数目:", len(inter_sec))


all_rm_guids = list(set(rm_hard_neg_guids + rm_hard_pos_guids))
# 4. 对原始数据进行筛选，删除用AUM找出来的hard negatives和hard positives
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/train-ds-samples.json', 'r') as f:
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
with open(f"/home/qi/Projects/NER/ds_ner/data/{dataset_name}/train-ds-samples-removed-noise-pos80.json", "w") as f:
        json.dump(new_ds, f)

# 4. 对原始数据进行筛选，标记用AUM找出来的hard negatives和hard positives
print("RM Positives #: ", len(rm_hard_pos_guids))
print("RM Negatives #: ", len(rm_hard_neg_guids))
print("RM Total #: ", len(all_rm_guids))
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/train-ds-samples.json', 'r') as f:
    ds_data = json.load(f)
# add a new key to each sample in ds_data
for i in trange(len(ds_data)):
    sample = ds_data[i]
    guids = sample['guids']
    spans_label = sample['spans_label']
    spans = sample['spans']
    span_status = []
    for j in range(len(guids)):
        guid = guids[j]
        if guid in all_rm_guids:
            span_status.append(0)
        else:
            span_status.append(1)
    ds_data[i]['span_status'] = span_status
# save the new ds_data
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset_name}/train-ds-samples-noise-flag-pos80.json', 'w') as f:
    json.dump(ds_data, f)
print("----- Saving new ds_data with noise-flag sucessfully!")