"""
Filtering and dataset mapping methods based on training dynamics.
By default, this module reads training dynamics from a given trained model and
computes the metrics---confidence, variability, correctness,
as well as baseline metrics of forgetfulness and threshold closeness
for each instance in the training data.
If specified, data maps can be plotted with respect to confidence and variability.
Moreover, datasets can be filtered with respect any of the other metrics.
"""
import argparse
import jsonx
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import tqdm
import json

from collections import defaultdict
from typing import List


def compute_correctness(trend: List[float]) -> float:
    """
    Aggregate #times an example is predicted correctly during all training epochs.
    """
    return sum(trend)

def compute_train_dy_metrics(training_dynamics, include_ci=False, burn_out=100):
    """
    Given the training dynamics (logits for each training instance across epochs), compute metrics
    based on it, for data map coorodinates.
    Computed metrics are: confidence, variability, correctness, forgetfulness, threshold_closeness---
    the last two being baselines from prior work
    (Example Forgetting: https://arxiv.org/abs/1812.05159 and
     Active Bias: https://arxiv.org/abs/1704.07433 respectively).
    Returns:
    - DataFrame with these metrics.
    - DataFrame with more typical training evaluation metrics, such as accuracy / loss.
    """
    confidence_ = {}
    variability_ = {}
    threshold_closeness_ = {}
    correctness_ = {}
    forgetfulness_ = {}

    # Functions to be applied to the data.
    variability_func = lambda conf: np.std(conf)
    if include_ci:    # Based on prior work on active bias (https://arxiv.org/abs/1704.07433)
        variability_func = lambda conf: np.sqrt(np.var(conf) + np.var(conf) * np.var(conf) / (len(conf)-1))
    threshold_closeness_func = lambda conf: conf * (1 - conf)

    loss = torch.nn.CrossEntropyLoss()

    num_tot_epochs = len(list(training_dynamics.values())[0]["logits"])
    print("Metrics computed: confidence, variability, correctness, threshold_closeness")

    logits = {i: [] for i in range(num_tot_epochs)}
    targets = {i: [] for i in range(num_tot_epochs)}
    training_accuracy = defaultdict(float)

    for guid in tqdm.tqdm(training_dynamics):
        correctness_trend = []
        true_probs_trend = []

        record = training_dynamics[guid]
        for i, epoch_logits in enumerate(record["logits"]):
            probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
            true_class_prob = float(probs[record["gold"]]) # 真实类别的Prob
            true_probs_trend.append(true_class_prob) # 真实类别的probs全部存到一个list里

            prediction = np.argmax(epoch_logits) # 最大的预测，即此次模型的预测值
            is_correct = (prediction == record["gold"]).item() # 这次是否正确
            correctness_trend.append(is_correct) # 存下每次是否正确。。

            training_accuracy[i] += is_correct
            logits[i].append(epoch_logits)
            targets[i].append(record["gold"])

        if burn_out < num_tot_epochs:
            correctness_trend = correctness_trend[:burn_out]
            true_probs_trend = true_probs_trend[:burn_out]

        correctness_[guid] = compute_correctness(correctness_trend) # 计算correctness
        confidence_[guid] = np.mean(true_probs_trend) # 平均的真实类别probs
        variability_[guid] = variability_func(true_probs_trend) # 平均在真实类别上的Varility

        # forgetfulness_[guid] = compute_forgetfulness(correctness_trend) # 用正确的趋势来计算forgetfulness
        threshold_closeness_[guid] = threshold_closeness_func(confidence_[guid])

    # Should not affect ranking, so ignoring.
    epsilon_var = np.mean(list(variability_.values()))

    column_names = ['guid',
                    'index',
                    'threshold_closeness',
                    'confidence',
                    'variability',
                    'correctness',
                    # 'forgetfulness',
                    ]
    df = pd.DataFrame([[guid,
                        i,
                        threshold_closeness_[guid],
                        confidence_[guid],
                        variability_[guid],
                        correctness_[guid],
                        # forgetfulness_[guid],
                        ] for i, guid in enumerate(correctness_)], columns=column_names)

    df_train = pd.DataFrame([[i,
                            loss(torch.Tensor(logits[i]), torch.LongTensor(targets[i])).item() / len(training_dynamics),
                            training_accuracy[i] / len(training_dynamics)
                            ] for i in range(num_tot_epochs)],
                        columns=['epoch', 'loss', 'train_acc'])
    return df, df_train
