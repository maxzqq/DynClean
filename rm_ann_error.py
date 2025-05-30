import json
import os
import sys
import numpy as np
import pandas as pd
from tqdm import trange

with open("/home/qi/Projects/NER/ds_ner/data/conll03/train-gs-samples.json", "r") as f:
    gs_data = json.load(f)
with open("/home/qi/Projects/NER/ds_ner/data/conll03/train-ds-samples.json", "r") as f:
    ds_data = json.load(f)

incomplete_guids = []
inaccurate_guids = []
error_guids = []
for i in range(len(gs_data)):
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

# 分离出ds数据中的entity span guids和非entity span guids
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
print("entity guid number and non-entity guid number:")
print(len(entity_guids), len(non_entity_guids))


# load training dynamics df
td_df = pd.read_json('/home/qi/Projects/NER/ds_ner/out/conll03/td_metrics.jsonl', lines=True)
# 拿到只有non-entity span guids的sorted_df
non_entity_td_df = td_df[td_df['guid'].isin(non_entity_guids)]
print(len(non_entity_td_df))

rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# sort by confidence
sorted_non_entity_td_df = non_entity_td_df.sort_values(by='confidence', ascending=True)

for rate in rates:
    print(f"Processing rate: {rate}......")
    selected_df = sorted_non_entity_td_df.head(n=int(rate * len(sorted_non_entity_td_df)))
    indices = list(selected_df['guid'])
    print(f"{rate}包含的span样本数目: ", len(indices))
    # 总的error guid
    error_guids = list(set(indices).intersection(incomplete_guids))
    print(len(error_guids))
    print(len(list(set(indices).intersection(inaccurate_guids))))

    with open("/home/qi/Projects/NER/ds_ner/data/conll03/train-ds-samples.json", "r") as f:
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
            if guids[j] in indices:
                new_guids.append(guids[j])
                new_spans_label.append(spans_label[j])
                nwe_spans.append(spans[j])
        ds_data[i]['guids'] = new_guids
        ds_data[i]['spans_label'] = new_spans_label
        ds_data[i]['spans'] = nwe_spans
        if len(ds_data[i]['spans']) > 0:
            new_ds.append(ds_data[i])
    # save the new ds_data
    print(f"New ds data length: {len(new_ds)}")
    with open(f"/home/qi/Projects/NER/ds_ner/data/conll03/train-ds-samples-top-{rate}-hard.json", "w") as f:
        json.dump(new_ds, f)