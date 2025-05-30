import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable

from allennlp.nn.util import batched_index_select
from allennlp.nn import util, Activation
from allennlp.modules import FeedForward

import numpy as np
from transformers import BertTokenizer, BertPreTrainedModel, BertModel
import os
import json
import pickle
import logging
from shared.const import PLMs
from sklearn.neighbors import NearestNeighbors
import numpy as np


logger = logging.getLogger('root')


class PLMsForEntity(nn.Module):
    def __init__(self, args, num_ner_labels, head_hidden_dim=150, width_embedding_dim=150, max_span_length=8):
        super(PLMsForEntity, self).__init__()

        self.args = args
        self.num_class = num_ner_labels
        self.PLM = PLMs[args.model]['model'].from_pretrained(args.model, output_hidden_states=False, return_dict=False)
        self.hidden_dropout = nn.Dropout(0.2) # 理解一下作用
        self.width_embedding = nn.Embedding(max_span_length+1, width_embedding_dim) # number of embeddings 为什么要加1？
        self.ner_classifier = nn.Sequential(
            FeedForward(input_dim=args.plm_hidden_size*2+width_embedding_dim, 
                        num_layers=2,
                        hidden_dims=head_hidden_dim,
                        activations=nn.ReLU(), #之前是F.relu
                        dropout=0.2),
        ) # input_dim: hidden_size*2+embedding_dim需要理解一下
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(head_hidden_dim, num_ner_labels)

        # load top hard guids
        # with open(args.top_hard_guids_path, 'rb') as f:
        #     self.top_hard_guids = pickle.load(f)
        # self.top_hard_guids = torch.tensor(self.top_hard_guids).to(torch.device("cuda:2"))



    def _get_span_embeddings(self, input_ids, spans, token_type_ids=None, attention_mask=None):
        """
        拿到一个batch的span embedding
        """
        # sequence_output, pooled_output, hidden_state 
        sequence_output, pooled_output = self.PLM(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = self.hidden_dropout(sequence_output)

        """
        spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(sequence_output, spans_start)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(sequence_output, spans_end)

        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.width_embedding(spans_width)

        spans_embedding = torch.cat((spans_start_embedding, spans_end_embedding, spans_width_embedding), dim=-1)
        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding

    def forward(self, input_ids, spans, spans_mask, spans_ner_label=None, spans_guid=None, token_type_ids=None, attention_mask=None, epoch=None):
        spans_embedding = self._get_span_embeddings(input_ids, spans, token_type_ids=token_type_ids, attention_mask=attention_mask) # 拿到一个batch的所有span embedding
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = self.fc(hidden)
        

        if spans_ner_label is not None:
            # Training 的时候，span_ner_label不为None
            loss_fct = CrossEntropyLoss(reduction='sum')

            # two loss functions for noise span
            loss_sce = SCELoss(num_classes=self.num_class, a=1, b=1)
            loss_gce = GCELoss(num_classes=self.num_class, q=0.3)

            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1 # 有label的span embedding, 应该是拿到有效的span embedding的index, size=1*batch_size*num_spans
                active_labels = torch.where(
                    active_loss, spans_ner_label.view(-1), torch.tensor(loss_fct.ignore_index).type_as(spans_ner_label)
                ) # 拿到有效的span embedding的label, size=1*batch_size*num_spans
                hidden_train = hidden.view(-1, hidden.shape[-1]) # 张量变换, size=1*batch_size*num_spans*embedding_dim

                # for guids
                active_guids = torch.where(
                    active_loss, spans_guid.view(-1), torch.tensor(loss_fct.ignore_index).type_as(spans_guid)
                ) # 拿到有效的span embedding的guid, size=1*batch_size*num_spans

                # postive hidden
                legal_index_pos = torch.nonzero((active_labels > 0)*1) # Get the index of the positive sample
                output_pos = hidden_train[legal_index_pos].squeeze(1)
                active_labels_pos = active_labels[legal_index_pos].squeeze(1).contiguous().view(-1, 1)

                # for positive guids
                active_guids_pos = active_guids[legal_index_pos].squeeze(1).contiguous().view(-1, 1)

                # negative hidden
                legal_index_neg = torch.nonzero((active_labels == 0 )*1)
                output_neg = hidden_train[legal_index_neg].squeeze(1)
                active_labels_neg = active_labels[legal_index_neg].squeeze(1).contiguous().view(-1, 1)

                # for negative guids
                active_guids_neg = active_guids[legal_index_neg].squeeze(1).contiguous().view(-1, 1)

                # # pos loss
                logits_pos = self.fc(output_pos)
                loss_pos = loss_fct(logits_pos, active_labels_pos.view(-1))

                
                if self.args.NO_NS:
                    # do nothing for negative loss
                    # original Distant Supervision
                    logits_neg = self.fc(output_neg)
                    loss_neg = loss_fct(logits_neg, active_labels_neg.view(-1))
                    active_labels_neg = active_labels_neg.view(-1)

                else:
                    # Negative Sampling
                    # neg loss
                    sim_score_neg1 = F.normalize(output_neg, dim=-1) @ F.normalize(output_pos, dim=-1).T 
                    sim_score_neg, idx_neg = torch.sort(torch.sum(sim_score_neg1, dim=-1), descending=False)
                    idx_neg2 = idx_neg[int(self.args.rate * len(idx_neg)):]
                
                    output_neg2 = output_neg[idx_neg2]
                    active_labels_neg2 = active_labels_neg[idx_neg2]

                    # for negative guids
                    active_guids_neg2 = active_guids_neg[idx_neg2].view(-1, 1) # 95%的负样本的guid。这里的guid是span的guid
                    active_guids_neg = active_guids_neg2

                    logits_neg = self.fc(output_neg2)
                    loss_neg = loss_fct(logits_neg, active_labels_neg2.view(-1))


                loss = loss_pos + loss_neg
                
            else:
                loss = loss_fct(logits.view(-1, logits.shape[-1]), spans_ner_label.view(-1))
            # return loss, logits, hidden, active_guids_neg2
            return loss, logits, hidden, active_guids_neg
        else:
            return logits, spans_embedding, hidden

class GCELoss(nn.Module):
    def __init__(self, num_classes, q=0.3):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()

class SCELoss(nn.Module):
    def __init__(self, num_classes=10, a=1, b=1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a #两个超参数
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CE 部分，正常的交叉熵损失
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0) #最小设为 1e-4，即 A 取 -4
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce.mean()
        return loss

class EntityModel():
    def __init__(self, args, num_ner_labels):
        super().__init__()

        bert_model_name = args.model
        vocab_name = bert_model_name
        
        if args.bert_model_dir is not None:
            bert_model_name = str(args.bert_model_dir) + '/'
            # vocab_name = bert_model_name + 'vocab.txt'
            vocab_name = bert_model_name
            logger.info('Loading PLMs model from {}'.format(bert_model_name))

        self.tokenizer = PLMs[args.model]['tokenizer'].from_pretrained(vocab_name, add_prefix_space=True)
        self.bert_model = PLMsForEntity(args, num_ner_labels=num_ner_labels, max_span_length=args.max_span_length)
        
        self._model_device = 'cpu'
        self.move_model_to_cuda()

    def move_model_to_cuda(self):
        if not torch.cuda.is_available():
            logger.error('No CUDA found!')
            exit(-1)
        logger.info('Moving to CUDA...')
        # self._model_device = 'cuda'
        self._model_device = torch.device("cuda:2")
        self.bert_model.to(self._model_device)
        # self.bert_model.cuda()
        # logger.info('# GPUs = %d'%(torch.cuda.device_count()))
        # if torch.cuda.device_count() > 1:
        #     self.bert_model = torch.nn.DataParallel(self.bert_model)

    def _get_input_tensors(self, tokens, spans, spans_ner_label, spans_guid):
        """
        Args:
        ------
        tokens: list of tokens
        spans: list of (start, end, span_index)
        spans_ner_label: list of labels
        spans_guid: list of guid
        ------
        Returns:
        ------
        tokens_tensor: torch.tensor of all the tokens, each element is the token id in the vocabulary
        spans_tensor: torch.tensor of the spans of one sentence, each element is [span_start, span_end, span_length]. Here the spans are the indices of the BERT tokens, which is different from the spans in the input (added special start token and sep token).
        spans_ner_label_tensor: torch.tensor of the spans' labels, each element is the label id
        """
        start2idx = []
        end2idx = []
        
        bert_tokens = []
        bert_tokens.append(self.tokenizer.cls_token) # 第一个bert_token的起始位置 [CLS]
        for token in tokens:
            start2idx.append(len(bert_tokens)) # 该token的起始idx
            sub_tokens = self.tokenizer.tokenize(' '+token) # " "似乎没啥用？
            bert_tokens += sub_tokens # sub_tokens是一个list, 直接相加
            end2idx.append(len(bert_tokens)-1) # 该token的结束idx
        bert_tokens.append(self.tokenizer.sep_token) # 最后一个bert_token的结束位置 [SEP]

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens) # bert_token转换为bert的token vocabulary id
        tokens_tensor = torch.tensor([indexed_tokens]) # 转换为tensor

        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans] # 转换为bert的span, span的起始和结束idx, span的长度 (原始token span的长度)
        bert_spans_tensor = torch.tensor([bert_spans])

        spans_ner_label_tensor = torch.tensor([spans_ner_label])
        
        spans_guid_tensor = torch.tensor([spans_guid])

        return tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_guid_tensor

    def _get_input_tensors_batch(self, samples_list, training=True):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        spans_guid_tensor_list = [] # 增加了spans_guid，和spans_ner_label一样的维度
        sentence_length = [] # 样本中token数量的list

        max_tokens = 0 # Batch样本中包含最多token的句子的token数量
        max_spans = 0 # Batch样本中包含最多span的句子的span数量
        for sample in samples_list:
            tokens = sample['tokens']
            spans = sample['spans']
            spans_ner_label = sample['spans_label']
            spans_guid = sample['guids'] # 增加了spans_guid

            tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_guid_tensor = self._get_input_tensors(tokens, spans, spans_ner_label, spans_guid)
            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)

            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            assert(bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1]) # 确保bert_spans_tensor和spans_ner_label_tensor的长度一致
            spans_guid_tensor_list.append(spans_guid_tensor)
            assert(bert_spans_tensor.shape[1] == spans_guid_tensor.shape[1]) # 确保bert_spans_tensor和spans_guid_tensor的长度一致
            


            if (tokens_tensor.shape[1] > max_tokens):
                # 样本中包含最多token的句子的token数量
                # 这个shape[1]是token的数量
                max_tokens = tokens_tensor.shape[1]
            if (bert_spans_tensor.shape[1] > max_spans):
                # 样本中包含最多span的句子的span数量
                # 这个shape[1]是span的数量
                max_spans = bert_spans_tensor.shape[1]
            sentence_length.append(sample['sent_length']) # 句子token数量
        sentence_length = torch.Tensor(sentence_length)

        # apply padding and concatenate tensors
        final_tokens_tensor = None
        final_attention_mask = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_guid_tensor = None # 增加了spans_guid
        final_spans_mask_tensor = None # span的mask？？？
        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_guid_tensor in zip(tokens_tensor_list, bert_spans_tensor_list, spans_ner_label_tensor_list, spans_guid_tensor_list):
            # padding for tokens
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens # 需要padding的长度
            attention_tensor = torch.full([1,num_tokens], 1, dtype=torch.long)
            if tokens_pad_length>0:
                # 需要padding
                pad = torch.full([1,tokens_pad_length], self.tokenizer.pad_token_id, dtype=torch.long) # 创建一个padding的tensor
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1) # 在tokens_tensor的最后一维上拼接pad
                attention_pad = torch.full([1,tokens_pad_length], 0, dtype=torch.long)
                attention_tensor = torch.cat((attention_tensor, attention_pad), dim=1)

            # padding for spans
            # ??? 为什么要padding spans
            num_spans = bert_spans_tensor.shape[1] # span的数量
            spans_pad_length = max_spans - num_spans # 需要padding的长度
            spans_mask_tensor = torch.full([1,num_spans], 1, dtype=torch.long)
            if spans_pad_length>0:
                # try:
                #     pad = torch.full([1,spans_pad_length,bert_spans_tensor.shape[2]], 0, dtype=torch.long) # shape[2]是span的数组的长度，也就是3
                # except:
                #     print(bert_spans_tensor.shape)
                #     print(spans_pad_length)
                #     print(spans_guid_tensor.shape)
                #     exit()

                pad = torch.full([1,spans_pad_length,3], 0, dtype=torch.long) # shape[2]是span的数组的长度，也就是3
                # pad = torch.full([1,spans_pad_length,bert_spans_tensor.shape[2]], 0, dtype=torch.long) # shape[2]是span的数组的长度，也就是3
                bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)
                mask_pad = torch.full([1,spans_pad_length], 0, dtype=torch.long) # mask_pad的shape是[1, spans_pad_length], 是2维的
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=1) # spans_mask_tensor的shape是[1, max_spans]
                spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=1)
                spans_guid_tensor = torch.cat((spans_guid_tensor, mask_pad), dim=1)
            # update final outputs
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor
                final_bert_spans_tensor = bert_spans_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_spans_guid_tensor = spans_guid_tensor
                final_spans_mask_tensor = spans_mask_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor,tokens_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, attention_tensor), dim=0)
                final_bert_spans_tensor = torch.cat((final_bert_spans_tensor, bert_spans_tensor), dim=0)

                final_spans_ner_label_tensor = torch.cat((final_spans_ner_label_tensor, spans_ner_label_tensor), dim=0)
                final_spans_guid_tensor = torch.cat((final_spans_guid_tensor, spans_guid_tensor), dim=0)

                final_spans_mask_tensor = torch.cat((final_spans_mask_tensor, spans_mask_tensor), dim=0)
        #logger.info(final_tokens_tensor)
        #logger.info(final_attention_mask)
        #logger.info(final_bert_spans_tensor)
        #logger.info(final_bert_spans_tensor.shape)
        #logger.info(final_spans_mask_tensor.shape)
        #logger.info(final_spans_ner_label_tensor.shape)
        # logger.info(final_spans_guid_tensor.shape)
        return final_tokens_tensor, final_attention_mask, final_bert_spans_tensor, final_spans_mask_tensor, final_spans_ner_label_tensor, final_spans_guid_tensor,sentence_length

    def run_batch(self, samples_list, epoch=None, try_cuda=True, training=True, dynamics=False):
        # convert samples to input tensors
        # tokens_tensor, attention_mask_tensor, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, sentence_length = self._get_input_tensors_batch(samples_list, training)
        tokens_tensor, attention_mask_tensor, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, spans_guid_tensor, sentence_length = self._get_input_tensors_batch(samples_list, training)

        # with open('./one_batch_data.pkl', 'wb') as f:
        #     pickle.dump([tokens_tensor, attention_mask_tensor, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, spans_guid_tensor, sentence_length], f)
        # exit()
        
        output_dict = {
            'ner_loss': 0,
        }

        if training:
            self.bert_model.train()
            # 如果需要进行筛选的话，那么这里要input guid
            ner_loss, ner_logits, spans_embedding, active_guids_neg = self.bert_model(
                input_ids = tokens_tensor.to(self._model_device),
                spans = bert_spans_tensor.to(self._model_device),
                spans_mask = spans_mask_tensor.to(self._model_device),
                spans_ner_label = spans_ner_label_tensor.to(self._model_device),
                spans_guid = spans_guid_tensor.to(self._model_device),
                attention_mask = attention_mask_tensor.to(self._model_device),
                epoch = epoch,
            )
            # input_ids, spans, spans_mask, spans_ner_label=None, spans_guid=None, token_type_ids=None, attention_mask=None, epoch=None
            # 存下每个guid的logits
            # with open('./logits.pkl', 'wb') as f:
            #     pickle.dump(ner_logits, f)
            # exit()
            
            output_dict['ner_loss'] = ner_loss.sum()
            output_dict['ner_llh'] = F.log_softmax(ner_logits, dim=-1)
            # 输出使用的negative span的guid
            output_dict['active_guids_neg'] = active_guids_neg
        else:
            self.bert_model.eval()
            with torch.no_grad():
                ner_logits, spans_embedding, last_hidden = self.bert_model(
                    input_ids = tokens_tensor.to(self._model_device),
                    spans = bert_spans_tensor.to(self._model_device),
                    spans_mask = spans_mask_tensor.to(self._model_device),
                    spans_ner_label = None,
                    attention_mask = attention_mask_tensor.to(self._model_device),
                )
            _, predicted_label = ner_logits.max(2)
            predicted_label = predicted_label.cpu().numpy()
            last_hidden = last_hidden.cpu().numpy()
            batch_logits = ner_logits.cpu().numpy()
            
            predicted = []
            pred_prob = []
            hidden = []
            pred_logits = [] # 存下logits
            # tensor_logits = [] # 存下原来的logits
            for i, sample in enumerate(samples_list):
                # each sample in one batch
                ner = []
                logits = []
                # ori_logits = []
                prob = []
                lh = []
                for j in range(len(sample['spans'])):
                    # each span in one sample
                    ner.append(predicted_label[i][j])
                    logits.append(batch_logits[i][j])
                    # ori_logits.append(ner_logits[i][j])
                    prob.append(F.softmax(ner_logits[i][j], dim=-1).cpu().numpy())
                    # prob.append(ner_logits[i][j].cpu().numpy())
                    lh.append(last_hidden[i][j])
                predicted.append(ner)
                pred_prob.append(prob)
                hidden.append(lh)
                pred_logits.append(logits)
                # tensor_logits.append(ori_logits)
            
            output_dict['pred_ner'] = predicted
            output_dict['ner_probs'] = pred_prob
            output_dict['ner_last_hidden'] = hidden
            output_dict['logits'] = pred_logits
            # output_dict['tensor_logits'] = tensor_logits
            


        return output_dict