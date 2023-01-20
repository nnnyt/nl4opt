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
from utils.reader_utils import extract_spans, get_tags, get_border_vocab_from_tag_vocab

from .modules import FocalLoss, PositionalEncoding
from .train_utils import save_wise_model, load_wise_model, get_pretrained_model_path

class CascadeNerModel(nn.Module):
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
        super(CascadeNerModel, self).__init__()
        self.tag_to_id = tag_to_id
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}

        self.border_to_id = get_border_vocab_from_tag_vocab(self.tag_to_id)
        self.id_to_border = {v: k for k, v in self.border_to_id.items()} # 用于标记BIO切分边界


        self.iob_tagging = iob_tagging
        self.tag_size = len(self.id_to_tag)
        self.border_size = len(self.id_to_border)

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
            # 分类预测头
            self.border_feedforward = nn.Linear(in_features=self.bert.config.hidden_size + self.pos_tag_embed_dim, out_features=self.border_size)
            # 类别预测
            self.tag_feedforward = nn.Linear(in_features=self.bert.config.hidden_size + self.pos_tag_embed_dim, out_features=self.tag_size)
        else:
            # 边界预测
            self.border_feedforward = nn.Linear(in_features=self.bert.config.hidden_size, out_features=self.border_size)
            # 类别预测
            self.tag_feedforward = nn.Linear(in_features=self.bert.config.hidden_size, out_features=self.tag_size)

        # 使用CRF层 ( 用于预测边界 )
        self.use_crf = use_crf
        self.border_crf_layer = ConditionalRandomField(num_tags=self.border_size, constraints=allowed_transitions(constraint_type="BIO", labels=self.id_to_border))
        
        self.tag_crf_layer = ConditionalRandomField(num_tags=self.tag_size, constraints=allowed_transitions(constraint_type="BIO", labels=self.id_to_tag))

        # 使用分类层-损失
        self.use_cls = use_cls
        self.use_wof = use_wof
        self.loss_weight = [1.0]*len(self.id_to_tag)
        # self.loss_weight[ self.tag_to_id["B-OBJ_NAME"] ] = 5.0
        # self.loss_weight[ self.tag_to_id["I-OBJ_NAME"] ] = 5.0
        # self.loss_weight[ self.tag_to_id["B-OBJ_NAME"] ] = 10.0
        # self.loss_weight[ self.tag_to_id["I-OBJ_NAME"] ] = 10.0
        # self.loss_weight[ self.tag_to_id["B-CONST_DIR"] ] = 2.0
        # self.loss_weight[ self.tag_to_id["I-CONST_DIR"] ] = 2.0

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
        self.border_f1 = SpanF1()
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
        tokens, tags, borders, mask, token_mask, metadata, pos_tag_idx, sentence_str = batch
        tokens, tags, borders, mask, token_mask = tokens.cuda(), tags.cuda(), borders.cuda(), mask.cuda(), token_mask.cuda()
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
            border_scores = self.border_feedforward(embeded_enhanced)
            tag_scores = self.tag_feedforward(embeded_enhanced)
        else:
            border_scores = self.border_feedforward(embedded_text_input)
            tag_scores = self.tag_feedforward(embedded_text_input)

        # project the token representation for classification
        border_scores = F.log_softmax(border_scores, dim=-1)
        tag_scores = F.log_softmax(tag_scores, dim=-1)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(border_scores=border_scores, tag_scores=tag_scores, mask=mask, tags=tags, borders=borders, metadata=metadata, batch_size=batch_size, mode=mode)
        return output

    def convert_tags(self, border_spans, pred_tag_scores):
        pred_tag_idxs = np.argmax(pred_tag_scores, -1).tolist()
        rec_tag_tokens = ["O"] * len(pred_tag_idxs)

        for span_range, tag in border_spans.items():
            if tag == self.border_to_id["O"]:
                continue
            
            s, e = span_range
            # 计数 获取 实体最终的标签
            span_tag_list = pred_tag_idxs[s: e+1]
            span_tag_list = [self.id_to_tag[t].replace("B-", "I-") for t in span_tag_list]

            best_tag = Counter(span_tag_list).most_common(1)[0][0] # [(tag, count)] -> tag

            for i in range(s, e+1):
                rec_tag_tokens[i] = best_tag
            rec_tag_tokens[s] = best_tag.replace("I-", "B-")
        return rec_tag_tokens

    def _compute_token_tags(self, border_scores, tag_scores, mask, tags, borders, metadata, batch_size, mode=''):
        # compute the log-likelihood loss and compute the best NER annotation sequence

        # 基于 CRF 预测 实体边界
        border_loss = -self.border_crf_layer(border_scores, borders, mask) / float(batch_size)
        best_border_path = self.border_crf_layer.viterbi_tags(border_scores, mask)

        # 计算 实体的分类损失
        if self.use_crf:
            tag_loss = -self.tag_crf_layer(tag_scores, tags, mask) / float(batch_size)
            best_tag_path = self.tag_crf_layer.viterbi_tags(tag_scores, mask)
        else:
            # （除去O） 
            active = torch.argmax(border_scores, -1).view(-1) > 0  # 根据边界取出为实体的部分
            active_tag_logits = tag_scores.view(-1, tag_scores.size()[2])[active]
            active_tag_labels = tags.view(-1)[active]
            # （不除去O）
            # active_tag_logits = tag_scores.view(-1, tag_scores.size()[2])
            # active_tag_labels = tags.view(-1)
            tag_flat_loss = self.criterion(active_tag_logits, active_tag_labels)
            tag_loss = torch.sum(tag_flat_loss) / float(batch_size) # tag_loss = torch.mean(tag_flat_loss) # / float(batch_size)

        pred_results, pred_tags = [], []
        pred_border_results, gold_border_results = [], []
        for i in range(batch_size):            
            pred_border_seq, _ = best_border_path[i]
            # 提取边界
            pred_border_spans = extract_spans([self.id_to_border[x] for x in pred_border_seq], iob_tagging=self.iob_tagging)
            pred_border_results.append(pred_border_spans)
            gold_border_results.append(
                extract_spans([self.id_to_border[b] for b in borders[i].cpu().detach().numpy()], iob_tagging=self.iob_tagging))

            if not self.use_crf:
                # 根据边界提取实体
                pred_tag_seq = self.convert_tags(
                    pred_border_spans, tag_scores[i].cpu().detach().numpy())
                tag_spans = extract_spans(pred_tag_seq, iob_tagging=self.iob_tagging)
            else:
                pred_tag_seq, _ = best_tag_path[i]
                tag_spans = extract_spans([self.id_to_tag[x] for x in pred_tag_seq], iob_tagging=self.iob_tagging)
            
            pred_tags.append(pred_tag_seq)
            pred_results.append(tag_spans)

        self.span_f1(pred_results, metadata)
        self.border_f1(pred_border_results, gold_border_results)
        
        output = {
            "border_loss": border_loss,
            "tag_loss": tag_loss,
            "results": self.span_f1.get_metric(),
            "border_results": self.border_f1.get_metric()
            }

        if mode == 'predict':
            output["predictions"] = {
                'token_tags': pred_tags,
                'pred_spans': pred_results,
                'gold_spans': metadata,
            }

        return output

    def predict_tags(self, batch, device='cuda:0'):
        tokens, tags, mask, token_mask, metadata, pos_tag_idx, sentence_str = batch
        tokens, mask, token_mask, tags = tokens.to(device), mask.to(device), token_mask.to(device), tags.to(device)

        pred_tags = self.forward(batch, mode='predict')['token_tags']
        tag_results = [compress(pred_tags_, mask_) for pred_tags_, mask_ in zip(pred_tags, token_mask)]
        return tag_results
