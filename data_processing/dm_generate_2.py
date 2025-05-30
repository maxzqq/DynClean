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

dataset = 'wiki'
print(f"----- Processing {dataset}!")

# load incomplete and inaccurate guids
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

# 1. 先处理Negatives
# 获取threshold：fake data 用于estimate threshold的aum/dm数据
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/aum/fake_class_guids.pkl', 'rb') as f:
    aum_guids = pickle.load(f)
fake_df = pd.read_json(f'/home/qi/Projects/NER/ds_ner/out/rb_{dataset}_aum/dy_log/td_df.jsonl', lines=True)
# fake_df = pd.read_json(f'/home/qi/Projects/NER/ds_ner/out/{dataset}_aum/dy_log/td_df.jsonl', lines=True)
neg_fake_df = fake_df[fake_df['guid'].isin(non_entity_guids)]
neg_fake_df = neg_fake_df[neg_fake_df['guid'].isin(aum_guids)]
neg_fake_df = neg_fake_df.sort_values(by=['confidence'], ascending=False)
thres = neg_fake_df.iloc[int(len(neg_fake_df)*0.1)]['confidence']

# 获得要删掉的hard negatives的guids
df = pd.read_json(f'/home/qi/Projects/NER/ds_ner/out/rb_{dataset}/dy_log/td_df.jsonl', lines=True)
# df = pd.read_json(f'/home/qi/Projects/NER/ds_ner/out/{dataset}/dy_log/td_df.jsonl', lines=True)
# 拿到只有non-entity span guids的sorted_df
neg_rm_guids = []
neg_df = df[df['guid'].isin(non_entity_guids)]
neg_df = neg_df.sort_values(by=['confidence'], ascending=True)
for i in range(len(neg_df)):
    row = neg_df.iloc[i]
    guid = row['guid']
    conf = row['confidence']
    if conf < thres:
        neg_rm_guids.append(guid)
print("Total RM #: ", len(neg_rm_guids))
inter_sec = list(set(neg_rm_guids).intersection(incomplete_guids))
print("RM False Negatives #: ", len(inter_sec))

# 2. 再处理Positives
# Confidence score
# fake data 用于estimate threshold的aum/dm数据
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/aum/fake_class_guids.pkl', 'rb') as f:
    aum_guids = pickle.load(f)
fake_df = pd.read_json(f'/home/qi/Projects/NER/ds_ner/out/rb_{dataset}_aum/dy_log/td_df.jsonl', lines=True)
# fake_df = pd.read_json(f'/home/qi/Projects/NER/ds_ner/out/{dataset}_aum/dy_log/td_df.jsonl', lines=True)
pos_fake_df = fake_df[fake_df['guid'].isin(entity_guids)]
pos_fake_df = pos_fake_df[pos_fake_df['guid'].isin(aum_guids)]
pos_fake_df = pos_fake_df.sort_values(by=['confidence'], ascending=False)
thres = pos_fake_df.iloc[int(len(pos_fake_df)*0)]['confidence']

# 获得要删掉的hard positives的guids
df = pd.read_json(f'/home/qi/Projects/NER/ds_ner/out/rb_{dataset}/dy_log/td_df.jsonl', lines=True)
# df = pd.read_json(f'/home/qi/Projects/NER/ds_ner/out/{dataset}/dy_log/td_df.jsonl', lines=True)
# 拿到只有non-entity span guids的sorted_df
pos_rm_guids = []
pos_df = df[df['guid'].isin(entity_guids)]
pos_df = pos_df.sort_values(by=['confidence'], ascending=True)
for i in range(len(pos_df)):
    row = pos_df.iloc[i]
    guid = row['guid']
    conf = row['confidence']
    if conf < thres:
        pos_rm_guids.append(guid)
print("Total RM #: ", len(pos_rm_guids))
inter_sec = list(set(pos_rm_guids).intersection(inaccurate_guids))
print("RM False Positives #: ", len(inter_sec))


all_rm_guids = list(set(neg_rm_guids + pos_rm_guids))
# 4. 对原始数据进行筛选，删除用AUM找出来的hard negatives和hard positives
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
with open(f"/home/qi/Projects/NER/ds_ner/data/{dataset}/train-ds-samples-removed-noise-DM.json", "w") as f:
        json.dump(new_ds, f)

# 4. 对原始数据进行筛选，标记用AUM找出来的hard negatives和hard positives
# print("RM Positives #: ", len(pos_rm_guids))
# print("RM Negatives #: ", len(neg_rm_guids))
# print("RM Total #: ", len(all_rm_guids))
# with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/train-ds-samples.json', 'r') as f:
#     ds_data = json.load(f)
# # add a new key to each sample in ds_data
# for i in trange(len(ds_data)):
#     sample = ds_data[i]
#     guids = sample['guids']
#     spans_label = sample['spans_label']
#     spans = sample['spans']
#     span_status = []
#     for j in range(len(guids)):
#         guid = guids[j]
#         if guid in all_rm_guids:
#             span_status.append(0)
#         else:
#             span_status.append(1)
#     ds_data[i]['span_status'] = span_status
# # save the new ds_data
# with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/train-ds-samples-noise-flag.json', 'w') as f:
#     json.dump(ds_data, f)
# print("----- Saving new ds_data with noise-flag sucessfully!")