import json
import argparse
import os
import sys
import random
import logging
import time
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import json
from shared.data_structures import Dataset
from shared.const import task_ner_labels, get_labelmap
from entity.utils import convert_dataset_to_samples, batchify, NpEncoder
from entity.models import EntityModel
import pickle

from transformers import AdamW, get_linear_schedule_with_warmup
import torch

from td_utilities import compute_train_dy_metrics
from aum import *

def compute_aum(TD_df, save_path):
    """
    Compute the aum score by given dataframe of each epoch
    """
    aum_calculator = AUMCalculator(save_path, compressed=False)
    for epoch in trange(len(TD_df)):
        df = TD_df[epoch]
        for index, row in tqdm(df.iterrows()):
            guid = row['guid']
            logit = torch.DoubleTensor(row['logits'])
            label = torch.LongTensor([row['gold']])
            # label_tensor = torch.LongTensor([label])
            # logit_tensor = torch.DoubleTensor(logit)
            aum_calculator.update(logit.unsqueeze(0), label.unsqueeze(0), (guid))
    aum_calculator.finalize()

def compute_td(TD_df, num_epochs):
    """
    Compute the training dynamics metrics
    """
    training_dynamics = {}
    for epoch in trange(num_epochs):
        df = TD_df[epoch]
        for index, row in df.iterrows():
            guid = row['guid']
            if guid not in training_dynamics:
                training_dynamics[guid] = {'gold': row['gold'], 'logits': []}
            training_dynamics[guid]['logits'].append(row['logits'])
    
    # total_epochs = len(list(training_dynamics.values())[0]["logits"])
    train_dy_metrics, _ = compute_train_dy_metrics(training_dynamics, num_epochs)
    return train_dy_metrics

td_df_path = '/home/qi/Projects/NER/ds_ner/out/new_wiki_bert_ns/dy_log'
target_epoch = 7
TD_df = []
print("====> Loading training dynamics!!!")
for epoch in trange(target_epoch+1):
    print(os.path.join(td_df_path, f"dynamics_epoch_{epoch}.jsonl"))
    df = pd.read_json(os.path.join(td_df_path, f"dynamics_epoch_{epoch}.jsonl"), lines=True)
    TD_df.append(df)
aum_save_path = f'/home/qi/Projects/NER/ds_ner/out/new_wiki_bert_ns/dy_log/aum_{target_epoch}'
if not os.path.exists(aum_save_path):
    os.makedirs(aum_save_path)
compute_aum(TD_df, aum_save_path)



