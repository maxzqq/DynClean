import pickle
import json
import os
from tqdm import tqdm, trange


dataset = 'twitter'
print('Cleaning the ds data:', dataset)
# load ds json data
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/train-ds-samples.json', 'r') as f:
    ds = json.load(f)
# load gs json data
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/train-gs-samples.json', 'r') as f:
    gs = json.load(f)


for i in trange(len(ds)):
    new_guids = []
    new_ds_spans = []
    new_ds_spans_label = []
    for j in range(len(ds[i]['spans_label'])):
        if ds[i]['spans_label'][j] == gs[i]['spans_label'][j]:
            new_ds_spans.append(ds[i]['spans'][j])
            new_ds_spans_label.append(ds[i]['spans_label'][j])
            new_guids.append(ds[i]['guids'][j])
    assert len(new_ds_spans) == len(new_ds_spans_label)
    assert len(new_ds_spans) == len(new_guids)
    ds[i]['spans'] = new_ds_spans
    ds[i]['spans_label'] = new_ds_spans_label
    ds[i]['guids'] = new_guids
# save the new ds data
with open(f'/home/qi/Projects/NER/ds_ner/data/{dataset}/train-ds-samples-filtered.json', 'w') as f:
    json.dump(ds, f)