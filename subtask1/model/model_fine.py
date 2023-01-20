from pickle import NONE
from re import L
from typing import List, Any
from itertools import accumulate, compress
import json
import os
import torch
from torch import nn
import torch.nn.functional as F
import copy
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions

from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModel, PretrainedConfig
from utils.metric import SpanF1
from utils.reader_utils import extract_spans, get_tags

from .modules import FocalLoss, PositionalEncoding
from .train_utils import save_wise_model, load_wise_model, get_pretrained_model_path
from .process_utils import post_process
from .model import NerModel


class FineNerModel(nn.Module):
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

                 pretrained_model_path=None,

                 trigger_to_idx = [],
                 
                 **argv
                 ):
        super(FineNerModel, self).__init__()
        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id

        self.iob_tagging = iob_tagging
        self.target_size = len(self.id_to_tag)

        self.pretrained_model = NerModel.from_pretrained(pretrained_model_path)
        self.freeze_pretrained_model()

        self.trigger_to_idx = trigger_to_idx
        self.encoder_model = encoder_model
        
        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict=True)
        
        if trigger_to_idx != []:
            # 增加 trigger 到 word_embedding 中
            self.len_with_tigger = max(list(trigger_to_idx.values())) + 1
            self.encoder.resize_token_embeddings(self.len_with_tigger)

        # 位置编码
        self.use_position = use_position
        if self.use_position:
            self.position_embedding = PositionalEncoding(d_model=self.encoder.config.hidden_size)

        # 词性特征
        # self.use_pos_tag = use_pos_tag
        # if self.use_pos_tag:
        #     self.pos_tag_to_idx = pos_tag_to_idx
        #     self.pos_tag_coprus_size = pos_tag_coprus_size
        #     self.pos_tag_pad_idx = pos_tag_pad_idx
        #     self.pos_tag_embed_dim = pos_tag_embed_dim if pos_tag_embed_dim is not None else self.encoder.config.hidden_size
        #     # 词性编码
        #     self.pos_tag_embeddings = nn.Embedding(num_embeddings=pos_tag_coprus_size,
        #                                            embedding_dim=self.pos_tag_embed_dim,
        #                                            padding_idx=pos_tag_pad_idx
        #                                         )
        #     # 分类预测头
        #     self.feedforward = nn.Linear(in_features=self.encoder.config.hidden_size + self.pos_tag_embed_dim, out_features=self.target_size)
        # else:
        #     # 分类预测头
        #     self.feedforward = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=self.target_size)

        # 使用CRF层
        self.use_crf = use_crf
        if self.use_crf:
            self.setup_fine_crf()

        # 使用分类层-损失
        self.use_cls = use_cls
        self.use_wof = use_wof
        if use_cls is not None:
            assert use_cls in ["crossentropy", "focal"]
            self.loss_weight = [1.0]*len(self.id_to_tag)
            # self.loss_weight[ self.tag_to_id["B-OBJ_NAME"] ] = 5.0
            # self.loss_weight[ self.tag_to_id["I-OBJ_NAME"] ] = 5.0
            # self.loss_weight[ self.tag_to_id["B-OBJ_NAME"] ] = 10.0
            # self.loss_weight[ self.tag_to_id["I-OBJ_NAME"] ] = 10.0
            # self.loss_weight[ self.tag_to_id["B-CONST_DIR"] ] = 2.0
            # self.loss_weight[ self.tag_to_id["I-CONST_DIR"] ] = 2.0
            self.loss_weight = torch.tensor(self.loss_weight, device="cuda:0")
            
            # 损失函数
            if use_cls == "crossentropy":
                # self.criterion = nn.CrossEntropyLoss()
                self.criterion = nn.NLLLoss(weight=self.loss_weight, reduction="none")
            elif use_cls == "focal":
                self.criterion = FocalLoss(weight=self.loss_weight, reduction="none")
        else:
            self.loss_weight = None
            self.criterion = None

        self.dropout = nn.Dropout(dropout_rate)

        # SpanF1 is required to evaluate the model - don't remove
        self.span_f1 = SpanF1()
        config = {k: v for k, v in locals().items() if k != "self" and k != "__class__" and k != "argv"}
        self.config = PretrainedConfig.from_dict(config)

    def freeze_pretrained_model(self):
        # 冻结参数
        for name, param in self.pretrained_model.named_parameters():
            param.requires_grad = False

    def setup_fine_crf(self):
        self.triggers = ["PARAM", "OBJ_DIR", "LIMIT"]
        self.fine_tag_to_id = dict()
        self.fine_tag_id_to_tag_id = dict()
        for k, v in self.tag_to_id.items():
            if k[2:] not in self.triggers:
                f_tag_id = len(self.fine_tag_to_id)
                self.fine_tag_to_id[k] = f_tag_id
                self.fine_tag_id_to_tag_id[ f_tag_id ] = v

        self.tag_id_to_fine_tag_id = {v: k for k, v in self.fine_tag_id_to_tag_id.items()}
        self.fine_id_to_tag = {v: k for k, v in self.fine_tag_to_id.items()}

        self.feedforward = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=len(self.fine_id_to_tag))

        self.crf_layer = ConditionalRandomField(num_tags=len(self.fine_id_to_tag), constraints=allowed_transitions(constraint_type="BIO", labels=self.fine_id_to_tag))
        

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

    def convert_adjust_batch(self, tokens, mask, tags, trigger_spans):
        tokens_with_trigger = tokens.cpu().detach().numpy().tolist()
        tags_with_trigger = tags.cpu().detach().numpy().tolist()
        mask_with_trigger = mask.cpu().detach().numpy().tolist()

        batch_size = len(mask_with_trigger)
        seq_max_len = len(mask_with_trigger[0])
        trigger_mask = [[0] * seq_max_len] * batch_size # 仅仅标记 B-trigger 的实体位置
        
        """ 初始化 tokens_with_trigger 等 """
        for i, spans in enumerate(trigger_spans):
            accu_count = 0
            for _span, _type in spans.items():
                s, e = _span
                # 获取 trigger idx 插入到输入序列中
                trigger = "trigger_" + _type.lower() # trigger_param
                trigger_type_idx = self.trigger_to_idx[ trigger ]
                
                insert_idx = s + accu_count

                tokens_with_trigger[i].insert(insert_idx, trigger_type_idx)
                trigger_mask[i].insert(insert_idx, 1)
                # 初始化
                mask_with_trigger[i].insert(insert_idx, 1)
                tags_with_trigger[i].insert(insert_idx, self.tag_to_id["O"]) # 原标签模式
                accu_count += 1
            tokens_with_trigger[i] = tokens_with_trigger[i][:seq_max_len]
            mask_with_trigger[i] = mask_with_trigger[i][:seq_max_len]
            tags_with_trigger[i] = tags_with_trigger[i][:seq_max_len]
            trigger_mask[i] = trigger_mask[i][:seq_max_len]
        
        """ 进一步修正 tags_with_trigger 对应的部分 """
        for seq_tags in tags_with_trigger:
            for i, _tag_id in enumerate(seq_tags):
                tag_label = self.id_to_tag[_tag_id][2:]
                if tag_label in (self.triggers + [""]):
                    # 将 tigger 和 O 对应的 实体标签 置为 O'
                    seq_tags[i] = self.fine_tag_to_id["O"] # 新标签模式
                else:
                    # BI 替换为 BI'
                    seq_tags[i] = self.tag_id_to_fine_tag_id[ _tag_id ] # 新标签模式

        tokens_with_trigger = torch.tensor(tokens_with_trigger).cuda()
        mask_with_trigger = torch.tensor(mask_with_trigger).cuda()
        tags_with_trigger = torch.tensor(tags_with_trigger).cuda()

        return tokens_with_trigger, mask_with_trigger, tags_with_trigger, trigger_mask

    def recover_seqs_from_trigger(self, seqs, trigger_mask, max_seq_len):
        # seqs = tags_with_trigger or tokens_with_trigger
        seqs_recover = []
        seqs_raw = seqs
        for i, seq in enumerate(seqs_raw):
            seq_r = []
            for j, item in enumerate(seq):
                # 过滤掉 trigger_mask 对应的部分
                if trigger_mask[i][j] == 1:
                    continue
                seq_r.append(item)
            seqs_recover.append(seq_r[:max_seq_len])
        
        return seqs_recover
    
    def forward(self, batch, mode='train', use_post=True):
        tokens, tags, mask, token_mask, metadata, pos_tag_idx, sentence_str = batch
        tokens, tags, mask, token_mask = tokens.cuda(), tags.cuda(), mask.cuda(), token_mask.cuda()
        if pos_tag_idx is not None:
            pos_tag_idx = pos_tag_idx.cuda()
        batch_size = tokens.size(0)

        # 使用 一阶段模型 的 trigger预测结果
        if False:
            self.pretrained_model.eval()
            pretrained_out = self.pretrained_model(batch, mode='predict', use_post=False)
            pred_spans = pretrained_out["predictions"]["pred_spans"]
            trigger_spans = []
            for spans in pred_spans:
                trigger_spans.append({
                    _span: _type for _span, _type in spans.items() if _type in self.triggers
                })
        else:
            # 使用 trigger 标签 
            trigger_spans = []
            for spans in metadata:
                trigger_spans.append({
                    _span: _type for _span, _type in spans.items() if _type in self.triggers
                })
        
        # 更正 输入 和 标签
        tokens_with_trigger, mask_with_trigger, tags_with_trigger, trigger_mask = self.convert_adjust_batch(tokens, mask, tags, trigger_spans)

        embedded_text_input = self.encoder(input_ids=tokens_with_trigger, attention_mask=mask_with_trigger.float())
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        if self.use_position:
            embedded_text_input =  self.position_embedding(embedded_text_input)

        # if self.use_pos_tag:
        #     embedded_pos_tag = self.pos_tag_embeddings(pos_tag_idx)
        #     embedded_pos_tag_ = embedded_pos_tag * mask.unsqueeze(2)
        #     embeded_enhanced = torch.cat((embedded_text_input, embedded_pos_tag_), dim=-1)
        #     token_scores = self.feedforward(embeded_enhanced)
        # else:

        token_scores = self.feedforward(embedded_text_input)
        # project the token representation for classification
        token_scores = F.log_softmax(token_scores, dim=-1)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(
            raw_tokens=tokens,
            tokens=tokens_with_trigger, token_scores=token_scores, mask=mask_with_trigger, tags=tags_with_trigger,
            trigger_mask=trigger_mask,
            trigger_spans=trigger_spans,
            metadata=metadata,
            
            batch_size=batch_size, mode=mode, use_post=use_post
            )
        return output

    def _compute_token_tags(self, raw_tokens, tokens, token_scores, mask, tags, trigger_mask, trigger_spans, metadata, batch_size, mode='train', use_post=True):
        # compute the log-likelihood loss and compute the best NER annotation sequence
        if self.use_crf:
            loss = -self.crf_layer(token_scores, tags, mask) / float(batch_size)
            best_path = self.crf_layer.viterbi_tags(token_scores, mask)
        else:
            assert self.use_cls is not None
            flat_loss = self.criterion(
                token_scores.view(-1, len(self.id_to_tag)), tags.view(-1))
            if self.use_wof:
                masks_of_entity = tags.view(-1) != self.tag_to_id["O"]
                weights_of_loss = masks_of_entity.float() + 0.5 # B  *S
                loss = torch.sum(flat_loss * weights_of_loss) / float(batch_size)
            else:
                loss = torch.sum(flat_loss) / float(batch_size)

            best_path = torch.argmax(token_scores, dim=-1).cpu().detach().numpy().tolist()

        pred_fine_tags = []
        for i in range(batch_size):
            if self.use_crf:
                fine_tag_seq, _ = best_path[i]
            else:
                assert self.use_cls is not None
                fine_tag_seq = best_path[i]
            
            pred_fine_tags.append([self.fine_id_to_tag[x] for x in fine_tag_seq])
        
        # （恢复）原始预测序列（除了trigger）
        pred_fine_spans = []
        # recover_tags = tags_trigger + tags_fine
        recover_fine_tags = self.recover_seqs_from_trigger(pred_fine_tags, trigger_mask, max_seq_len=raw_tokens.shape[1])
        for seq in recover_fine_tags:
            pred_fine_spans.append(extract_spans(seq, iob_tagging=self.iob_tagging))
  
        pred_results = []
        # 合并 第一阶段 和 第二阶段 的预测结果
        for x, y in zip(trigger_spans, pred_fine_spans):
            spans = copy.deepcopy(y)
            spans.update(x)
            pred_results.append(spans)

        # 后处理
        if use_post:
            pred_results = post_process(pred_results, raw_tokens.cpu().detach().numpy().tolist())

        self.span_f1(pred_results, metadata)
        output = {"loss": loss, "results": self.span_f1.get_metric()}

        if mode == 'predict':
            output["predictions"] = {
                'token_tags': recover_fine_tags,
                'pred_spans': pred_results,
                'gold_spans': metadata,
            }

        return output

    def predict_tags(self, batch, use_post=True):
        tokens, tags, mask, token_mask, metadata, pos_tag_idx, sentence_str = batch
        
        pred_tags = self.forward(batch, mode='predict', use_post=use_post)['predictions']['token_tags']
        tag_results = [compress(pred_tags_, mask_) for pred_tags_, mask_ in zip(pred_tags, token_mask)]
        return tag_results
