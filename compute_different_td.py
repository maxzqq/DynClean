
from td_utilities import compute_train_dy_metrics
from tqdm import trange
import pandas as pd
import numpy as np
import json


# 计算各种数量不同的training dynamics
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
