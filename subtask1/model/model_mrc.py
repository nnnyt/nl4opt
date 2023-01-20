from multiprocessing.sharedctypes import Value
from pickle import NONE
from typing import List, Any
from itertools import compress
import json
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import Counter
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions

from transformers import get_linear_schedule_with_warmup, AutoModel, PretrainedConfig
from utils.metric import SpanF1
from utils.reader_utils import extract_spans, get_tags

from .modules import FocalLoss, PositionalEncoding
from .train_utils import save_wise_model, load_wise_model, get_pretrained_model_path


class MRCNerModel(nn.Module):
    def __init__(self,
                 dropout_rate=0.1,
                 tag_to_id=None,
                 iob_tagging="conll",
                 encoder_model='xlm-roberta-base',

                 use_pos_tag=False, 
                 pos_tag_to_idx={},
                 pos_tag_coprus_size=None,
                 pos_tag_embed_dim=None,
                 pos_tag_pad_idx=0,

                 use_position=False,

                 use_crf=True,
                 use_cls=None,
                 use_wof=False,
                 
                 **argv
                 ):
        super(MRCNerModel, self).__init__()
        self.tag_to_id = tag_to_id
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}

        self.iob_tagging = iob_tagging
        self.tag_size = len(self.id_to_tag)

        self.encoder_model = encoder_model
        self.bert = AutoModel.from_pretrained(encoder_model, return_dict=True)

        # 位置编码
        self.use_position = use_position
        if self.use_position:
            self.position_embedding = PositionalEncoding(d_model=self.bert.config.hidden_size)

        # 词性特征
        self.use_pos_tag = use_pos_tag
        if self.use_pos_tag:
            self.pos_tag_to_idx = pos_tag_to_idx
            self.pos_tag_coprus_size = pos_tag_coprus_size
            self.pos_tag_pad_idx = pos_tag_pad_idx
            self.pos_tag_embed_dim = pos_tag_embed_dim if pos_tag_embed_dim is not None else self.bert.config.hidden_size
            # 词性标签
            self.pos_tag_embeddings = nn.Embedding(num_embeddings=pos_tag_coprus_size,
                                                   embedding_dim=self.pos_tag_embed_dim,
                                                   padding_idx=pos_tag_pad_idx
                                                )
            # 开始预测
            self.start_feedforward = nn.Linear(in_features=self.bert.config.hidden_size + self.pos_tag_embed_dim, out_features=2)
            # 结束预测
            self.end_feedforward = nn.Linear(in_features=self.bert.config.hidden_size + self.pos_tag_embed_dim, out_features=2)
        else:
            # 开始预测
            self.start_feedforward = nn.Linear(in_features=self.bert.config.hidden_size, out_features=2)
            # 结束预测
            self.end_feedforward = nn.Linear(in_features=self.bert.config.hidden_size, out_features=2)

        # 使用分类层-损失
        self.use_cls = use_cls
        self.use_wof = use_wof
        self.loss_weight = [1.0, 1.0]
        self.loss_weight = torch.tensor(self.loss_weight)

        self.loss_weight = self.loss_weight.cuda()
        if use_cls == "crossentropy":
            self.criterion = nn.NLLLoss(weight=self.loss_weight, reduction="none")
        elif use_cls == "focal":
            self.criterion = FocalLoss(weight=self.loss_weight, reduction="none")
        else:
            raise ValueError("use_cls is illegal !")
    
        self.dropout = nn.Dropout(dropout_rate)

        # SpanF1 is required to evaluate the model - don't remove
        self.span_f1 = SpanF1()
        config = {k: v for k, v in locals().items() if k != "self" and k != "__class__" and k != "argv"}
        self.config = PretrainedConfig.from_dict(config)
    
    def save_config(self, config_dir):
        self.config.save_pretrained(config_dir)

    @classmethod
    def from_config(cls, config_path, **argv):
        with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            model_config.update(argv)
            return cls(
                **model_config
            )

    @classmethod
    def from_pretrained(cls, model_dir, **argv):
        config_path = os.path.join(model_dir, "config.json")
        model = cls.from_config(config_path, **argv)

        model_path = get_pretrained_model_path(model_dir)
        model = load_wise_model(model, model_path)
        
        return model

    def save_pretrained(self, model_dir, filename):
        out_model_path = os.path.join(model_dir, filename)
        save_wise_model(self, out_model_path)
        self.save_config(model_dir)

    def forward(self, batch, mode='', **argv):
        tokens, starts, ends, mask, types, token_mask, metadata, pos_tag_idx, sentence_str = batch
        tokens, starts, ends, mask, types, token_mask = tokens.cuda(), starts.cuda(), ends.cuda(), mask.cuda(), types.cuda(), token_mask.cuda()
        if pos_tag_idx is not None:
            pos_tag_idx = pos_tag_idx.cuda()
        batch_size = tokens.size(0)

        embedded_text_input = self.bert(input_ids=tokens, attention_mask=mask.float())
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        if self.use_position:
            embedded_text_input =  self.position_embedding(embedded_text_input)

        if self.use_pos_tag:
            embedded_pos_tag = self.pos_tag_embeddings(pos_tag_idx)
            embedded_pos_tag_ = embedded_pos_tag * mask.unsqueeze(2)
            embeded_enhanced = torch.cat((embedded_text_input, embedded_pos_tag_), dim=-1)
            start_scores = self.start_feedforward(embeded_enhanced)
            end_scores = self.end_feedforward(embeded_enhanced)
        else:
            start_scores = self.start_feedforward(embedded_text_input)
            end_scores = self.end_feedforward(embedded_text_input)

        # project the token representation for classification
        start_scores = F.log_softmax(start_scores, dim=-1)
        end_scores = F.log_softmax(end_scores, dim=-1)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(start_scores=start_scores, end_scores=end_scores, mask=mask, start_labels=starts, end_labels=ends, type_labels=types, metadata=metadata, batch_size=batch_size, mode=mode)
        return output

    
    # 严格解码 baseline
    def mrc_decode(self, start_preds, end_preds, ent_type, mask=None):
        '''返回实体的start, end
        '''
        predict_entities = []

        if mask is not None: # 预测的把query和padding部分mask掉
            start_preds = torch.argmax(start_preds, -1) * mask
            end_preds = torch.argmax(end_preds, -1) * mask

        start_preds = start_preds.cpu().numpy()
        end_preds = end_preds.cpu().numpy()

        predict_tags = []
        for bt_i in range(start_preds.shape[0]):
            start_pred = start_preds[bt_i]
            end_pred = end_preds[bt_i]
            tag_seq = []
            span_set = dict()
            # 统计每个样本的结果
            for i, s_type in enumerate(start_pred):
                if s_type == 0:
                    tag_seq.append("O")
                    continue
                for j, e_type in enumerate(end_pred[i:]):
                    if s_type == e_type:
                        # [样本id, 实体起点，实体终点，实体类型]
                        ent_idx = ent_type[bt_i].item()
                        ent_tag = self.id_to_tag[ ent_idx ]
                        ent_tag = ent_tag[2:] if ent_tag != "O" else "O"
                        
                        span_set[(i, i+j)] = ent_tag
                        tag_seq.append( ent_tag )
                        break
            predict_tags.append(tag_seq)
            predict_entities.append(span_set)
        return predict_tags, predict_entities


    def _compute_token_tags(self, start_scores, end_scores, mask, start_labels, end_labels, type_labels, metadata, batch_size, mode=''):
        # compute the log-likelihood loss and compute the best NER annotation sequence

        # 计算实体的 开始位置损失
        start_flat_loss = self.criterion(
            start_scores.view(-1, start_scores.size()[2]), start_labels.view(-1))
        start_loss = torch.sum(start_flat_loss) / float(batch_size)

        # 计算实体的 开始位置损失
        end_flat_loss = self.criterion(
            end_scores.view(-1, end_scores.size()[2]), end_labels.view(-1))
        end_loss = torch.sum(end_flat_loss) / float(batch_size)

        loss = start_loss + end_loss

        pred_tags, pred_spans = self.mrc_decode(start_scores, end_scores, type_labels, mask=mask)

        self.span_f1(pred_spans, metadata)
        
        output = {
            "loss": loss,
            "start_loss": start_loss,
            "end_loss": end_loss,
            "results": self.span_f1.get_metric(),
            }

        if mode == 'predict':
            output["predictions"] = {
                'token_tags': pred_tags,
                'pred_spans': pred_spans,
                'gold_spans': metadata,
            }

        return output

    def predict_tags(self, batch, device='cuda:0'):
        tokens, tags, mask, token_mask, metadata, pos_tag_idx, sentence_str = batch
        tokens, mask, token_mask, tags = tokens.to(device), mask.to(device), token_mask.to(device), tags.to(device)

        pred_tags = self.forward(batch, mode='predict')['token_tags']
        tag_results = [compress(pred_tags_, mask_) for pred_tags_, mask_ in zip(pred_tags, token_mask)]
        return tag_results
