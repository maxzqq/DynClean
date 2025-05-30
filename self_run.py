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

def gen_pseudo_label(args, model, batches,tot_gold):
    """
    Generate the pseudo labels for the training set and evaluate the model on training dataset.
    Save the pseudo labels to the output directory as pickle file and the key is guid.
    """
    logger.info('Generating pseudo labels...')
    cor = 0
    tot_pred = 0
    l_cor = 0
    l_tot = 0
    wrong_span = 0
    pseudo_labels = {}
    for i in trange(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            guids = sample['guids']
            for guid, gold, pred in zip(guids, sample['spans_label'], preds):
                l_tot += 1
                if pred == gold:
                    l_cor += 1
                if pred != 0 and gold != 0 and pred == gold:
                    cor += 1
                if pred != 0:
                    tot_pred += 1
                    if gold == 0:
                        wrong_span += 1
                # no matter what, save the pseudo labels
                pseudo_labels[guid] = pred
    acc = l_cor / l_tot
    logger.info('wrong spans count: %d'%(wrong_span))
    logger.info('Accuracy: %5f'%acc)
    logger.info('Cor: %d, Pred TOT: %d, Gold TOT: %d'%(cor, tot_pred, tot_gold))
    p = cor / tot_pred if cor > 0 else 0.0
    r = cor / tot_gold if cor > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    logger.info('P: %.5f, R: %.5f, F1: %.5f'%(p, r, f1))
    # save the pseudo labels
    with open(args.output_dir+str('/pseudo_labels.p'), 'wb') as f:
        pickle.dump(pseudo_labels, f)
    logger.info('Pseudo labels saved to %s'%(args.output_dir+str('/pseudo_labels.p')))
    

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


def separate_samples(samples):
    """
    Separate the samples into entity and non-entity
    """
    entity_guids = []
    non_entity_guids = []
    for sample in samples:
        guids = sample['guids']
        spans_label = sample['spans_label']
        for i in range(len(spans_label)):
            if spans_label[i] == 0:
                non_entity_guids.append(guids[i])
            else:
                entity_guids.append(guids[i])
    return entity_guids, non_entity_guids

def td_filter(samples, td_path, rate):
    """
    Removing the samples with low confidence
    """
    # get entity_guids and non_entity_guids
    entity_guids, non_entity_guids = separate_samples(samples)
    total_span_samples = len(entity_guids) + len(non_entity_guids)
    # load the training dynamics
    td_df = pd.read_json(td_path, lines=True)
    # sort the dataframe by the confidence
    sorted_df = td_df.sort_values(by='confidence', ascending=True)
    # 拿到只有非entity samples的dataframe
    non_entity_sorted_df = sorted_df[sorted_df['guid'].isin(non_entity_guids)]
    # select the samples with the lowest confidence
    selected_df = non_entity_sorted_df.head(n=int(rate * len(non_entity_sorted_df)))
    # get the indices of the selected samples
    indices = list(selected_df['guid'])
    # filter the span samples by the indices
    logger.info("Starting filtering the training samples by training dynmaics...")
    legal_span_num = 0
    new_samples = []
    for i in trange(len(samples)):
        sample = samples[i]
        guids = sample['guids']
        spans = sample['spans']
        spans_label = sample['spans_label']
        new_guids = []
        new_spans = []
        new_spans_label = []
        for j in range(len(spans_label)):
            if guids[j] not in indices:
                new_guids.append(guids[j])
                new_spans.append(spans[j])
                new_spans_label.append(spans_label[j])
        samples[i]['guids'] = new_guids
        samples[i]['spans'] = new_spans
        samples[i]['spans_label'] = new_spans_label
        legal_span_num += len(new_spans)
    true_rate = (total_span_samples - legal_span_num) / total_span_samples
    logger.info("Before filtering, the number of span samples is %d"%total_span_samples)
    logger.info("After filtering, the number of legal span samples is %d"%legal_span_num)
    logger.info("The number of filtered span samples is %d"%(total_span_samples - legal_span_num))
    logger.info("Expected removing ratio: %.4f"%rate)
    logger.info("Actual removing ratio: %.4f"%true_rate)
    return samples

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default=None, required=True, choices=['conll03', 'ontonotes', 'bc5cdr',  'twitter', 'wiki', 'webpage', 'onto', 'onto2'])

    parser.add_argument('--data_dir', type=str, default=None, required=True, 
                        help="path to the preprocessed dataset")               
    parser.add_argument('--output_dir', type=str, default='entity_output', 
                        help="output directory of the entity model")

    # 这个参数对结果影响大吗？
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
    parser.add_argument('--eval_per', type=float, default=.5, 
                        help="how often evaluating the trained model on dev set during training")
    parser.add_argument("--bertadam", action="store_true", help="If bertadam, then set correct_bias = False")
    parser.add_argument('--test_pred_filename', type=str, default="ent_pred_test.json", help="the prediction filename for the test set")

    parser.add_argument('--do_train', action='store_true', 
                        help="whether to run training")
    parser.add_argument('--do_generate', action='store_true', 
                        help="whether to run training")
    parser.add_argument('--train_shuffle', action='store_true',
                        help="whether to train with randomly shuffled data")
    parser.add_argument('--do_eval', action='store_true', 
                        help="whether to run evaluation")
    parser.add_argument('--eval_test', action='store_true', 
                        help="whether to evaluate on test set")

    parser.add_argument('--model', type=str, default='bert-base-uncased', 
                        help="the base model name (a huggingface model)")
    parser.add_argument('--bert_model_dir', type=str, default=None, 
                        help="the base model directory")
    parser.add_argument('--plm_hidden_size', type=int, default=768)                  
    parser.add_argument('--rate', type=float, default=0.95, help='samplinng rate for negative sampling')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dm', action='store_true', 
                        help="whether to record the dataset catography training dynamics")
    parser.add_argument('--pos_rate', type=float, default=0.02, help='Positives pruning rate')
    parser.add_argument('--neg_rate', type=float, default=0.001, help='Negaatives pruning rate')

    parser.add_argument('--dm_filter', action='store_true', 
                        help="whether to record the dataset catography training dynamics")
    parser.add_argument('--no_filter', action='store_true', 
                        help="whether to record the dataset catography training dynamics")
    parser.add_argument('--dm_rate', type=float, default=0.05, help='samplinng rate for negatives')
    
    parser.add_argument('--dm_stat', action='store_true', 
                        help="whether to record the dataset catography training dynamics")

    parser.add_argument('--aum', action='store_true', 
                        help="whether to record the area under margin training dynamics")
    
    parser.add_argument('--save_ns', action='store_true', 
                        help="whether to record the sampling negtive samples guid")
    
    parser.add_argument('--eval_train', action='store_true', 
                        help="Evaulate the model on the training set")
    
    parser.add_argument('--dev_perform', action='store_true', 
                        help="Store the performance of the model on the dev set")
    
    parser.add_argument('--data_mode', type=str, default='TD-NS', required=True, choices=['TD-NS-DS', 'NS-DS', 'TD', 'DS', 'GS','Clean-DS'])

    parser.add_argument('--NO_NS', action='store_true', 
                        help="Without Negative Sampling")
    
    parser.add_argument('--base_model', type=str, default='roberta_ns', required=False, choices=['roberta_ns', 'roberta', 'bert', 'bert_ns'])

    args = parser.parse_args()

    # set up dataset paths
    args.train_data = os.path.join(args.data_dir, 'train-ds.json')
    args.dev_data = os.path.join(args.data_dir, 'dev.json')
    args.test_data = os.path.join(args.data_dir, 'test.json')

    # set up span-level dataset paths
    if args.data_mode == 'GS':
        logger.info('Using Gold-Standard samples...')
        args.train_samples = os.path.join(args.data_dir, 'train-gs-samples.json')
    elif args.data_mode == 'Clean-DS':
        logger.info('Using Clean DS samples...')
        args.train_samples = os.path.join(args.data_dir, 'clean-train-ds-samples.json')
    else:
        # load the distant supervision samples
        logger.info('Using DS samples...')
        args.train_samples = os.path.join(args.data_dir, 'train-ds-samples.json')
    args.dev_samples = os.path.join(args.data_dir, 'dev-samples.json')
    args.test_samples = os.path.join(args.data_dir, 'test-samples.json')

    # add gs sample path
    args.gs_samples = os.path.join(args.data_dir, 'gs-samples.json')

    # set up training dynamics saving paths
    if args.dm:
        args.dm_path = os.path.join(args.output_dir, 'dy_log')
        if not os.path.exists(args.dm_path):
            os.makedirs(args.dm_path)
    args.ns_save_path = os.path.join(args.output_dir, f'ns_guids/{args.data_mode}')
    if not os.path.exists(args.ns_save_path):
        os.makedirs(args.ns_save_path)

    # args.top_hard_guids_path = os.path.join(args.data_dir, 'top_hard_guids.pkl')

    setseed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # set up logger
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))

    logger.info(sys.argv)
    logger.info(args)

    # --- for store the performance on dev set ---
    if args.dev_perform:
        dev_perform_path = args.output_dir + '/dev_perform'
        if not os.path.exists(dev_perform_path):
            os.makedirs(dev_perform_path)

     # ---- for show # of noise negative during training -----
    if args.save_ns:
        incom_path = os.path.join(args.data_dir, 'incomplete.pkl')
        with open(incom_path, 'rb') as f:
            incomplete = pickle.load(f)
    
    # if args.dm_filter:
    #     td_path = os.path.join(args.output_dir, 'td_metrics.jsonl')
    
    ner_label2id, ner_id2label = get_labelmap(task_ner_labels[args.task])
    
    num_ner_labels = len(task_ner_labels[args.task]) + 1 # +1 for non-entities (0)
    model = EntityModel(args, num_ner_labels=num_ner_labels)

    print(ner_label2id)

    # convert dataset to samples, 可以省略掉。这里保留是为了和原来的代码保持一致；
    # dev_ner是一个整数，表示dev集中的实体数量，用于计算F1值
    dev_samples, dev_ner = convert_dataset_to_samples(args.dev_data, args.max_span_length, ner_label2id=ner_label2id, training=False)

    # directly load samples
    dev_samples = json.load(open(args.dev_samples))
    
    # batchify samples
    dev_batches = batchify(dev_samples, args.eval_batch_size)
    # convert train dataset to samples
    train_samples, train_ner = convert_dataset_to_samples(args.train_data, args.max_span_length, ner_label2id=ner_label2id, training=True)
    

    if args.do_train:

        # directly load samples
        train_samples = json.load(open(args.train_samples))

        if args.dm_filter:
            logger.info("===>Loading the training dynamics filtered dataset ...")
            # train_samples = td_filter(train_samples, td_path, args.dm_rate)
            new_train_samples_path = os.path.join(args.data_dir, f'{args.base_model}/train-ds-samples-removed-noise.json')
            # new_train_samples_path = os.path.join(args.data_dir, 'train-ds-samples-removed-noise.json')
            # with open(new_train_samples_path, 'w') as f:
            #     json.dump(train_samples, f)
            with open(new_train_samples_path, 'r') as f:
                train_samples = json.load(f)
            logger.info("Used datasets: %s"%new_train_samples_path)

        # print(train_samples[:5])
        train_batches = batchify(train_samples, args.train_batch_size)
        best_result = 0.0
        best_epoch = 0

        # for dev performance string
        dev_performance = []

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
        t_total = len(train_batches) * args.num_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total*args.warmup_proportion), t_total)
        
        tr_loss = 0
        tr_examples = 0
        global_step = 0
        eval_step = int(len(train_batches) * args.eval_per)

        TD_df = [] # saving the TD dataframe of each epoch

        for _ in tqdm(range(args.num_epoch)):
            neg_guids = [] # for storing the negative guids
            
            epoch_eval = {
                'epoch': [], 'global_step': [], 'f1': [], 'p': [], 'r': []
            } # 如何用dataclass来实现这个功能？

            if args.train_shuffle:
                # shuffle the training batches
                random.shuffle(train_batches)
            for i in trange(len(train_batches)):
                output_dict = model.run_batch(train_batches[i], training=True)
                loss = output_dict['ner_loss']
                loss.backward()

                tr_loss += loss.item()
                tr_examples += len(train_batches[i])
                global_step += 1

                optimizer.step()
                optimizer.zero_grad()

                if global_step % args.print_loss_step == 0:
                    logger.info('Epoch=%d, iter=%d, loss=%.5f'%(_, i, tr_loss / tr_examples))
                    tr_loss = 0
                    tr_examples = 0
                
                if global_step % eval_step == 0:
                    # print('Performance on train,,,,')
                    # f1 = evaluate(model, train_batches, train_ner)
                    # print('Performance on dev ... ')
                    f1, p, r = evaluate(model, dev_batches, dev_ner)
                    epoch_eval['epoch'].append(_)
                    epoch_eval['global_step'].append(global_step)
                    epoch_eval['f1'].append(f1)
                    epoch_eval['p'].append(p)
                    epoch_eval['r'].append(r)
                    
                    if args.save_ns:
                        # print(">>>>==== Containts False Negative #:", len(list(set(neg_guids)&set(incomplete))))
                        print(">>>>==== Containts False Negative #:", inter_length(neg_guids, incomplete))

                    if f1 > best_result:
                        best_result = f1
                        best_epoch = _
                        logger.info('!!! Best valid (epoch=%d): %.2f' % (_, f1*100))
                        save_model(model, args)

                # record the negative guids
                if args.save_ns:
                    active_guids_neg = output_dict['active_guids_neg'].cpu().numpy()
                    active_guids_neg = active_guids_neg.flatten().tolist()
                    neg_guids = neg_guids + active_guids_neg


            # After one epoch, record the training dynamics
            if args.dm:
                # with open('./train_batches.pkl', 'wb') as f:
                #     pickle.dump(train_batches, f)
                iter_df = dm_evalate(model, train_batches, _, args.dm_path)
                TD_df.append(iter_df)
            # at the end of each epoch, saving the stored negative guids
            if args.save_ns:
                print(">>>>==== Containts False Negative #:", inter_length(neg_guids, incomplete))
                with open(args.ns_save_path + f'/ns_neg_guids_{_}.pkl', 'wb') as f:
                    pickle.dump(neg_guids, f)
            
            # After one epoch, record the dev performance epoch_eval
            if args.dev_perform:
                dev_performance.append(epoch_eval)
            
            # 每次train完，对整个training set也进行一次evalutate
            # print(f'Performance on train of {_} =====>')
            # f1 = evaluate(model, train_batches, train_ner)
            # print('Performance on dev ... ')
        # after training, save the dev performance
        if args.dev_perform:
            file_name =  args.data_mode + '.json'
            with open(dev_perform_path + '/' + file_name, 'w') as f:
                json.dump(dev_performance, f)

    if args.do_generate:
        model = EntityModel(args, num_ner_labels=num_ner_labels)
        model.bert_model.load_state_dict(torch.load(args.output_dir+str('/best_model.m')), strict=False)
        if args.dm_filter:
            logger.info("===>Loading the training dynamics filtered dataset ...")
            # train_samples = td_filter(train_samples, td_path, args.dm_rate)
            new_train_samples_path = os.path.join(args.data_dir, f'{args.base_model}/train-ds-samples-removed-noise.json')
            # new_train_samples_path = os.path.join(args.data_dir, 'train-ds-samples-removed-noise.json')
            # with open(new_train_samples_path, 'w') as f:
            #     json.dump(train_samples, f)
            with open(new_train_samples_path, 'r') as f:
                train_samples = json.load(f)
            logger.info("Used datasets: %s"%new_train_samples_path)
        if args.no_filter:
            # use original training data
            logger.info("===>Loading the original dataset ...")
            new_train_samples_path = os.path.join(args.data_dir, 'train-ds-samples.json')
            with open(new_train_samples_path, 'r') as f:
                train_samples = json.load(f)
            logger.info("Used datasets: %s"%new_train_samples_path)
        train_batches = batchify(train_samples, 256)
        logger.info('Generating Pseudo Labels ===>')
        gen_pseudo_label(args, model, train_batches, train_ner)
        logger.info('Pseudo Labels Generated ===>')
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

    if args.do_eval:

        model = EntityModel(args, num_ner_labels=num_ner_labels)
        model.bert_model.load_state_dict(torch.load(args.output_dir+str('/best_model.m')), strict=False)

        if args.dm_filter:
            logger.info("===>Loading the training dynamics filtered dataset ...")
            # train_samples = td_filter(train_samples, td_path, args.dm_rate)
            new_train_samples_path = os.path.join(args.data_dir, f'{args.base_model}/train-ds-samples-removed-noise.json')
            # new_train_samples_path = os.path.join(args.data_dir, 'train-ds-samples-removed-noise.json')
            # with open(new_train_samples_path, 'w') as f:
            #     json.dump(train_samples, f)
            with open(new_train_samples_path, 'r') as f:
                train_samples = json.load(f)
            logger.info("Used datasets: %s"%new_train_samples_path)

        # print(train_samples[:5])
        train_batches = batchify(train_samples, args.train_batch_size)


        if args.do_train:
            logger.info('Best Dev Performance ... at [epoch] %d'%best_epoch)
        evaluate(model, dev_batches, dev_ner)

        if args.eval_train:
            logger.info('Evaluating Train ===>')
            evaluate(model, train_batches, train_ner)
        
        # if args.pseudo_label:
            

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
    # After training, 
    # 1. computing and saving the Confidence training dynamics
    if args.dm:
        logger.info("====> Confidence computing!!!")
        train_dy_metrics  = compute_td(TD_df, args.num_epoch)
        train_dy_filename = os.path.join(args.dm_path, "td_df.jsonl")
        train_dy_metrics.to_json(train_dy_filename,
                            orient='records',
                            lines=True)
        logger.info(f"Confidence based on Training Dynamics written to {train_dy_filename}")
        # 2. computing and saving the AUM training dynamics
        logger.info("====> AUM computing!!!")
        # compute_aum(TD_df, args.dm_path)
        for epoch_num in range(10, args.num_epoch+1):
            print(f"====> AUM computing for epoch {epoch_num}")
            aum_save_path = os.path.join(args.dm_path, f"aum_{epoch_num}")
            # create the aum_save_path folder
            if not os.path.exists(aum_save_path):
                os.makedirs(aum_save_path)
            # compute the aum
            compute_aum(TD_df[:epoch_num], aum_save_path)
        logger.info(f"AUM metrics written to {args.dm_path}")


