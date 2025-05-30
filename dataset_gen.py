import json
import os
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import pickle
import torch
from torch.nn import CrossEntropyLoss
import pickle


# 分离出ds数据中的entity span guids和非entity span guids
with open("/home/qi/Projects/NER/ds_ner/data/conll03/train-ds-samples.json", "r") as f:
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
print("----- Loading non_entity_guids sucessfully!")

# load training dynamics df
td_df = pd.read_json('/home/qi/Projects/NER/ds_ner/out/conll03/ns_td_metrics_3epoch.jsonl', lines=True)
# 拿到只有non-entity span guids的sorted_df
non_entity_td_df = td_df[td_df['guid'].isin(non_entity_guids)]
# non_entity_td_df = td_df
print(len(non_entity_td_df))

# sort by confidence
sorted_non_entity_td_df = non_entity_td_df.sort_values(by='confidence', ascending=True)

rates = [0.1]
for rate in rates:

    selected_df = sorted_non_entity_td_df.head(n=int(rate * len(sorted_non_entity_td_df)))
    indices = list(selected_df['guid'])
    print(f"{rate}包含的span样本数目: ", len(indices))

    with open("/home/qi/Projects/NER/ds_ner/data/conll03/train-ds-samples-top-remove-0.3-inaccurate.json", "r") as f:
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
            if guids[j] not in indices:
                new_guids.append(guids[j])
                new_spans_label.append(spans_label[j])
                nwe_spans.append(spans[j])
        ds_data[i]['guids'] = new_guids
        ds_data[i]['spans_label'] = new_spans_label
        ds_data[i]['spans'] = nwe_spans
        if len(ds_data[i]['spans']) > 0:
            new_ds.append(ds_data[i])
    print(f"New ds data length: {len(new_ds)}")
    with open(f"/home/qi/Projects/NER/ds_ner/data/conll03/train-ds-samples-top-remove-0.3-inaccurate-{rate}-incom.json", "w") as f:
            json.dump(new_ds, f)