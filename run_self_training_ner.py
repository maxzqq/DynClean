# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import argparse
import os
import sys
import copy
import random
import logging
import time
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import json
from shared.data_structures import Dataset
from shared.const import task_ner_labels, get_labelmap
from entity.utils import convert_dataset_to_samples, batchify, NpEncoder, soft_frequency, multi_source_label_refine
from entity.models import EntityModel
import pickle

from transformers import AdamW, get_linear_schedule_with_warmup
import torch

# from td_utilities import compute_train_dy_metrics
# from aum import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')

def save_model(model, args):
    """
    Save the model to the output directory
    """
    logger.info('Saving model to %s...'%(args.output_dir))
    torch.save(model.bert_model.state_dict(), args.output_dir+str('/best_model.m'))

def output_ner_predictions(model, batches, dataset, output_file):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    start_result = {}
    end_result = {}
    span_hiddens = []
    hard_neg_hiddens = []
    tot_pred_ett = 0
    k = 0
    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        pred_hidden = output_dict['ner_last_hidden']
        pred_probs = output_dict['ner_probs']
        span_hidden = []
        hard_neg_hidden = []
        for sample, preds, hiddens, probs in zip(batches[i], pred_ner, pred_hidden, pred_probs):
            off = 0
            ner_result[str(k)] = []
            span_hid = []
            hard_neg_hid = []
            for span, pred, hidden, prob in zip(sample['spans'], preds, hiddens, probs):
                if pred == 0:
                    if max(prob) < 0.95:
                        hard_neg_hid.append(hidden)
                    continue
                ner_result[str(k)].append([span[0]+off, span[1]+off, ner_id2label[pred]])
                span_hid.append(hidden)
            span_hidden.append(span_hid)
            hard_neg_hidden.append(hard_neg_hid)
            k += 1
        span_hiddens.extend(span_hidden)
        hard_neg_hiddens.extend(hard_neg_hidden)

    print(len(ner_result))
    print(ner_result[str(1)])
    tot_pred_ett = len(list(ner_result.values()))
    with open('bb_ner_result.json', 'w') as f:
        json.dump(ner_result, f)
    with open('hidden.npy', 'wb') as f:
        np.save(f, span_hiddens)    
    with open('hard_neg_hidden.npy', 'wb') as f:
        np.save(f, hard_neg_hiddens)   
    logger.info('Total pred entities: %d'%tot_pred_ett)


def evaluate(model, batches, tot_gold):
    """
    Evaluate the entity model
    """
    logger.info('Evaluating...')
    cor = 0
    tot_pred = 0
    l_cor = 0
    l_tot = 0
    wrong_span = 0

    for i in trange(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            for gold, pred in zip(sample['spans_label'], preds):
                l_tot += 1
                if pred == gold:
                    l_cor += 1
                if pred != 0 and gold != 0 and pred == gold:
                    cor += 1
                if pred != 0:
                    tot_pred += 1
                    if gold == 0:
                        wrong_span += 1
                   
    acc = l_cor / l_tot
    logger.info('wrong spans count: %d'%(wrong_span))
    logger.info('Accuracy: %5f'%acc)
    logger.info('Cor: %d, Pred TOT: %d, Gold TOT: %d'%(cor, tot_pred, tot_gold))
    p = cor / tot_pred if cor > 0 else 0.0
    r = cor / tot_gold if cor > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    logger.info('P: %.5f, R: %.5f, F1: %.5f'%(p, r, f1))
    print('Teach model performance --- P: %.5f, R: %.5f, F1: %.5f'%(p, r, f1))

    return f1, p, r

def dm_evalate(model, batches, epoch, saving_path):
    """
    Evaluate the entity model and record the dataset catography training dynamics.
    """
    logger.info('------ Recording Training Dynamics (Epoch %s) -----------'%epoch)

    all_guid = []
    all_gold = []
    all_logits = []

    for i in trange(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        logits_list = output_dict['logits']
        for sample, logits in zip(batches[i], logits_list):
            guid_list = sample['guids']
            label_list = sample['spans_label']
            for j in range(len(guid_list)):
                all_guid.append(guid_list[j])
                all_gold.append(label_list[j])
                all_logits.append(logits[j])
            # for guid, label, logit in zip(guid_list, label_list, logits):
            #     # if guid in all_guids:
            #     #     continue
            #     all_guids.append(guid)
            #     record = {
            #         'guid': guid,
            #         'logits_epoch_%s'%epoch: logit,
            #         'gold': label,
            #         'device':'cuda:2'
            #     }
            #     training_dynamics.append(record)
        # break
    logger.info(f'---- Num of training_dynamics: {len(all_guid)}')
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    
    # convert into dataframe
    df = pd.DataFrame({
        'guid': all_guid,
        'gold': all_gold,
        f'logits_{epoch}': all_logits
    })
    # save to jsonl
    df.to_json(saving_path + f'/dynamics_epoch_{epoch}.jsonl', orient='records', lines=True)
    logger.info(f'Epoch {epoch} Saved to [{saving_path}]')
    logger.info(f'Epoch {epoch} Processed!!!')

    df = pd.DataFrame({
        'guid': all_guid,
        'gold': all_gold,
        'logits': all_logits
    })

    return df


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


def inter_length(list1, list2):
    """
    Get the length of intersection of two lists
    """
    return len(list(set(list1).intersection(set(list2))))

def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_model(args, num_ner_labels):
    """
    Initialize the model
    """
    model = EntityModel(args, num_ner_labels=num_ner_labels)
    if args.model_name_or_path:
        model.bert_model.load_state_dict(torch.load(args.model_name_or_path))
    return model

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default=None, required=True, choices=['conll03', 'bc5cdr',  'twitter', 'wiki', 'webpage', 'onto', 'onto2'])
    parser.add_argument('--output', type=str, default='self_out', 
                        help="output directory of the entity model")            

    parser.add_argument('--max_span_length', type=int, default=8, 
                        help="spans w/ length up to max_span_length are considered as candidates")
    
    parser.add_argument('--train_batch_size', type=int, default=16, 
                        help="batch size during training")
    parser.add_argument('--eval_batch_size', type=int, default=32, 
                        help="batch size during inference")
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                        help="learning rate for the BERT encoder")
    parser.add_argument('--task_learning_rate', type=float, default=5e-4, 
                        help="learning rate for task-specific parameters, i.e., classification head")
    parser.add_argument('--warmup_proportion', type=float, default=0, 
                        help="the ratio of the warmup steps to the total steps")
    parser.add_argument('--num_epoch', type=int, default=5, 
                        help="number of the training epochs")
    parser.add_argument('--num_iteration', type=int, default=5, 
                        help="number of the noise filter iterations")
    parser.add_argument('--print_loss_step', type=int, default=100, 
                        help="how often logging the loss value during training")
    
    parser.add_argument("--bertadam", action="store_true", help="If bertadam, then set correct_bias = False")
    parser.add_argument('--test_pred_filename', type=str, default="ent_pred_test.json", help="the prediction filename for the test set")

    parser.add_argument('--do_train', action='store_true', 
                        help="whether to run training")
    parser.add_argument('--train_shuffle', action='store_true',
                        help="whether to train with randomly shuffled data")
    parser.add_argument('--do_eval', action='store_true', 
                        help="whether to run evaluation")
    parser.add_argument('--eval_test', action='store_true', 
                        help="whether to evaluate on test set")

    parser.add_argument('--model', type=str, default='roberta-base', 
                        help="the base model name (a huggingface model)")
    parser.add_argument('--bert_model_dir', type=str, default=None, 
                        help="the base model directory")
    parser.add_argument('--plm_hidden_size', type=int, default=768)                  
    parser.add_argument('--rate', type=float, default=0.95, help='samplinng rate for negative sampling')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dm', action='store_true', 
                        help="whether to record the dataset catography training dynamics")

    parser.add_argument('--dm_filter', action='store_true', 
                        help="whether to record the dataset catography training dynamics")

    parser.add_argument('--aum', action='store_true', 
                        help="whether to record the area under margin training dynamics")
    
    parser.add_argument('--save_ns', action='store_true', 
                        help="whether to record the sampling negtive samples guid")
    
    parser.add_argument('--eval_train', action='store_true', 
                        help="Evaulate the model on the training set")
    
    parser.add_argument('--dev_perform', action='store_true', 
                        help="Store the performance of the model on the dev set")
    
    parser.add_argument('--data_mode', type=str, default='DS', required=True, choices=['DSST', 'DS'])

    parser.add_argument('--NO_NS', action='store_true', 
                        help="No Negative Sampling")
    
    parser.add_argument('--eval_per', type=float, default=.5, 
                        help="how often evaluating the trained model on dev set during training")
    
    parser.add_argument('--model_name_or_path', type=str, 
                        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.")
    parser.add_argument('--self_training_begin_step', type = int, default = 900, help = 'the begin step (usually after the first epoch) to start self-training.')
    parser.add_argument('--self_training_label_mode', type = str, default = "soft", help = 'pseudo label type. choices:[hard(default), soft].')
    parser.add_argument('--self_training_period', type = int, default = 1800, help = 'the self-training period.')
    parser.add_argument('--self_training_hp_label', type = float, default = 5.9, help = 'use high precision label.')




    args = parser.parse_args()
    args.data_dir = os.path.join('data', args.task)
    args.output_dir = os.path.join(args.output, args.task)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # the trained model
    args.model_name_or_path = os.path.join('out_td', args.task)
    args.model_name_or_path += str('/best_model.m')

    # original token-level data files
    args.train_data = os.path.join(args.data_dir, 'train-ds.json')
    args.dev_data = os.path.join(args.data_dir, 'dev.json')
    args.test_data = os.path.join(args.data_dir, 'test.json')

    # set up span-level dataset paths
    if args.data_mode == 'GS':
        logger.info('Using Gold-Standard samples...')
        args.train_samples = os.path.join(args.data_dir, 'train-gs-samples.json')
    elif args.data_mode == 'Clean-DS':
        logger.info('Using Clean DS samples...')
        args.train_samples = os.path.join(args.data_dir, 'train-ds-samples-filtered.json')
    else:
        # load the distant supervision samples
        logger.info('Using DS samples...')
        args.train_samples = os.path.join(args.data_dir, 'train-ds-samples.json')
    args.dev_samples = os.path.join(args.data_dir, 'test-samples.json')
    args.test_samples = os.path.join(args.data_dir, 'test-samples.json')

    setseed(args.seed)
    
    # set up logger
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(sys.argv)
    logger.info(args)
    
    ner_label2id, ner_id2label = get_labelmap(task_ner_labels[args.task])
    num_ner_labels = len(task_ner_labels[args.task]) + 1 # +1 for non-entities (0)
    print(ner_label2id)

    # convert dataset to samples, 可以省略掉。这里保留是为了和原来的代码保持一致；
    # dev_ner是一个整数，表示dev集中的实体数量，用于计算F1值
    dev_samples, dev_ner = convert_dataset_to_samples(args.dev_data, args.max_span_length, ner_label2id=ner_label2id, training=False)
    # directly load samples
    dev_samples = json.load(open(args.dev_samples))
    # batchify samples
    dev_batches = batchify(dev_samples, args.eval_batch_size)

    # initialize the model
    model = EntityModel(args, num_ner_labels=num_ner_labels)
    if os.path.exists(args.model_name_or_path):
        # load the trained model
        model.bert_model.load_state_dict(torch.load(args.model_name_or_path))
        trained_step = 900 # for conll03
        logger.info('====>>> Loaded trained model from %s'%args.model_name_or_path)
        print("evaluating...")
        logger.info('====>>> Evaluated the trained model on dev set')
        evaluate(model, dev_batches, dev_ner)
    
    if args.do_train:
        # convert train dataset to samples
        # 同dev_samples， train_ner是一个整数，表示train集中的实体数量，用于计算F1值
        train_samples, train_ner = convert_dataset_to_samples(args.train_data, args.max_span_length, ner_label2id=ner_label2id, training=True)

        # directly load samples
        train_samples = json.load(open(args.train_samples))
        logger.info("Used datasets: %s"%args.train_samples)

        train_batches = batchify(train_samples, args.train_batch_size)
        best_result = 0.0
        best_epoch = 0

        # set up the optimizer
        # 这是initial model的一部分？
        param_optimizer = list(model.bert_model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                if 'PLM' in n],
            'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer
                if 'PLM' not in n], 'lr': args.task_learning_rate,
            'weight_decay': 0.0}
                ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=not(args.bertadam))

        # computing the total steps
        t_total = len(train_batches) * args.num_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total*args.warmup_proportion), t_total)
        
        tr_loss = 0
        tr_examples = 0
        global_step = trained_step # for conll03
        logger.info(f'====>>> Global Step starts at {global_step} ...')
        eval_step = int(len(train_batches) * args.eval_per)
        
        self_training_teacher_model = model # initial teacher model

        for _ in tqdm(range(args.num_epoch)):
            # neg_guids = [] # for storing the negative guids
            epoch_eval = {
                'epoch': [], 'global_step': [], 'f1': [], 'p': [], 'r': []
            } # 如何用dataclass来实现这个功能？
            if args.train_shuffle:
                # shuffle the training batches
                random.shuffle(train_batches)
            for i in range(len(train_batches)):

                # update labels periodically after certain begin step
                if global_step >= args.self_training_begin_step:
                    # logger.info('===>Self-Training begins at step %d'%args.self_training_begin_step)
                    # 这里开始使用self-training生成伪标签
                    # Update a new teacher periodically
                    delta = global_step - args.self_training_begin_step # 参数T3, 即update teacher的周期
                    if delta % args.self_training_period == 0:
                        logger.info('===> Updating Teacher Model at step %d !!!!'%global_step)
                        self_training_teacher_model = copy.deepcopy(model) # deep copy the current model as the teacher
                        logger.info('====>>> Evaluated the Teacher Model on dev set')
                        f1, p, r = evaluate(self_training_teacher_model, dev_batches, dev_ner)
                        # evaluate(self_training_teacher_model, dev_batches, dev_ner)
                    
                    # Using the teacher model to generate pseudo labels
                    output_dict = self_training_teacher_model.run_batch(train_batches[i], training=False)
                    # save as pickle file
                    # save_pickle(output_dict, f'teacher_output.pkl')

                    label_mask = None
                    if args.self_training_label_mode == 'hard':
                        # just generate the hard labels
                        pass
                    elif args.self_training_label_mode == 'soft':
                        # generate the soft labels
                        pred_labels = soft_frequency(logits=output_dict['tensor_logits'], power=2)
                        pred_labels, label_mask = multi_source_label_refine(args=args, pred_labels=pred_labels)
                        # print(pred_labels.shape)
                        # print(label_mask.shape)
                        # # print(pred_labels[0][0])
                        # print(label_mask[0])
                        # exit()
                    
                    # update the training batch with the pseudo labels
                    # save_pickle(train_batches[i], f'train_batch_{i}.pkl')
                    # save_pickle((pred_labels, label_mask), f'pseudo_labels_{i}.pkl')
                    # exit()

                output_dict = model.run_batch(samples_list=train_batches[i], pseudo_labels=pred_labels, label_mask=label_mask ,training=True)
                loss = output_dict['ner_loss']

                loss.backward()

                tr_loss += loss.item()
                tr_examples += len(train_batches[i])
                

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.print_loss_step == 0:
                    logger.info('Epoch=%d, iter=%d, loss=%.5f'%(_, global_step, tr_loss / tr_examples))
                    tr_loss = 0
                    tr_examples = 0
                
                if global_step % eval_step == 0:
                    logger.info('Performance on dev ... ')
                    f1, p, r = evaluate(model, dev_batches, dev_ner)
                    epoch_eval['epoch'].append(_)
                    epoch_eval['global_step'].append(global_step)
                    epoch_eval['f1'].append(f1)
                    epoch_eval['p'].append(p)
                    epoch_eval['r'].append(r)

                    if f1 > best_result:
                        best_result = f1
                        best_epoch = _
                        logger.info('!!! Best valid (epoch=%d): %.2f' % (_, f1*100))
                        save_model(model, args)
        # Training Done
        logger.info('Training Done!')
        logger.info('Best Dev Performance ... at [epoch] %d'%best_epoch)
        # saving the epoch_eval as json
        with open(os.path.join(args.output_dir, 'epoch_eval.json'), 'w') as f:
            json.dump(epoch_eval, f, cls=NpEncoder)

    if args.do_eval:

        model = EntityModel(args, num_ner_labels=num_ner_labels)
        model.bert_model.load_state_dict(torch.load(args.output_dir+str('/best_model.m')))
        if args.do_train:
            logger.info('Best Dev Performance ... at [epoch] %d'%best_epoch)
        evaluate(model, dev_batches, dev_ner)

        if args.eval_train:
            logger.info('Evaluating Train ===>')
            evaluate(model, train_batches, train_ner)

        logger.info('Evaluating test...')


        # convert test dataset to samples
        test_samples, test_ner = convert_dataset_to_samples(args.test_data, args.max_span_length, ner_label2id=ner_label2id)

        # directly load samples
        test_samples = json.load(open(args.test_samples))

        test_batches = batchify(test_samples, args.eval_batch_size)


        # train_samples, train_ner = convert_dataset_to_samples(args.train_data, args.max_span_length, ner_label2id=ner_label2id, training=True)
        # # directly load samples
        # train_samples = json.load(open(args.train_samples))
        # # print(train_samples[:5])
        # train_batches = batchify(train_samples, args.train_batch_size)
        # print("best performance on training set: =====")

        # evaluate(model, train_batches, train_ner)

        evaluate(model, test_batches, test_ner)
        # prediction_file = os.path.join(args.output_dir, args.test_pred_filename)
        # output_ner_predictions(model, test_batches, args.test_data, output_file=prediction_file)


