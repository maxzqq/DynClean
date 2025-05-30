import json
import os
from tqdm import tqdm,trange
import numpy as np
import pandas as pd
import pickle
import torch
from torch.nn import CrossEntropyLoss
import pickle

# load train-ds data
with open('/home/qi/Projects/NER/ds_ner/data/bc5cdr/train-ds-samples.json', 'r') as f:
    ds_data = json.load(f)
with open('/home/qi/Projects/NER/ds_ner/data/bc5cdr/train-gs-samples.json', 'r') as f:
    gs_data = json.load(f)

incomplete_guids = []
inaccurate_guids = []
error_guids = []
for i in range(len(gs_data)):
    # 确保guids和tokens一致
    assert gs_data[i]['tokens'] == ds_data[i]['tokens']
    assert gs_data[i]['guids'] == ds_data[i]['guids']
    
    ds_guids = ds_data[i]['guids']
    ds_spans_label = ds_data[i]['spans_label']
    gs_spans_label = gs_data[i]['spans_label']

    assert len(ds_guids) == len(gs_spans_label)

    for j in range(len(ds_spans_label)):
        ds_label = ds_spans_label[j]
        gs_label = gs_spans_label[j]
        guid = ds_guids[j]
        if ds_label != gs_label:
            error_guids.append(guid)
            if ds_label == 0 and gs_label != 0:
                # ds中没有标注成ner，但是gs中标注成了ner
                incomplete_guids.append(guid)
            else:
                inaccurate_guids.append(guid)
print(len(error_guids), len(incomplete_guids), len(inaccurate_guids))
print("----- Loading incomplete_guids and inaccurate_guids sucessfully!")

with open('/home/qi/Projects/NER/ds_ner/data/bc5cdr/train-ds-samples.json', 'r') as f:
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


# load training dynamics df
td_df = pd.read_json('/home/qi/Projects/NER/ds_ner/out/bc5cdr/td_metrics_3epoch.jsonl', lines=True)
non_entity_td_df = td_df[td_df['guid'].isin(non_entity_guids)]
# non_entity_td_df = td_df
print(len(non_entity_td_df))

# sort by confidence
sorted_non_entity_td_df = non_entity_td_df.sort_values(by='confidence', ascending=True)

rate = 0.05
selected_df = sorted_non_entity_td_df.head(n=int(rate * len(sorted_non_entity_td_df)))
indices = list(selected_df['guid'])
print(f"{rate}包含的span样本数目: ", len(indices))
# 总的error guid
rm_error_guids = list(set(indices).intersection(error_guids))
print("删掉的错误样本数目:", len(rm_error_guids))
print("------ Loading top hard guid indices sucessfully!")

for idx in range(3):
    print(f"------- Testing {idx} -----------")
    with open(f'/home/qi/Projects/NER/ds_ner/out/bc5cdr/ns_guids/ns_neg_guids_{idx}.pkl', 'rb') as f:
        neg_guids = pickle.load(f)

    print("没取差集的 better sampling中 error数目:", len(list(set(neg_guids).intersection(incomplete_guids))))

    top_sim_low_confidence_guids = list(set(neg_guids).intersection(indices))
    print("Better Sampling和Top Hard的交集: ", len(top_sim_low_confidence_guids))
    print("交集中的Error数目: ", len(list(set(top_sim_low_confidence_guids).intersection(incomplete_guids))))
    print("交集中的非Error数目: ", len(top_sim_low_confidence_guids) - len(list(set(top_sim_low_confidence_guids).intersection(incomplete_guids))))
    # remove top_sim_low_confidence_guids from neg_guids_0
    neg_guids_0_rm_hard = list(set(neg_guids).difference(top_sim_low_confidence_guids))
    print("和top hard取差集后的error:", len(list(set(neg_guids_0_rm_hard).intersection(incomplete_guids))))