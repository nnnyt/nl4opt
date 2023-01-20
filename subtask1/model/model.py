from itertools import compress
import json
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions

from transformers import get_linear_schedule_with_warmup, AutoModel, AutoTokenizer, PretrainedConfig
from utils.metric import SpanF1
from utils.reader_utils import extract_spans, get_tags

from .modules import FocalLoss, PositionalEncoding
from .train_utils import save_wise_model, load_wise_model, get_pretrained_model_path
from .process_utils import post_process

class NerModel(nn.Module):
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
                 tongxiao_post=True,
                 
                 use_lstm=False,
                 gpu="cuda:0",
                 **argv
                 ):
        super(NerModel, self).__init__()
        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id

        self.iob_tagging = iob_tagging
        self.target_size = len(self.id_to_tag)

        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.tongxiao_post = tongxiao_post

        # 位置编码
        self.use_position = use_position
        if self.use_position:
            self.position_embedding = PositionalEncoding(d_model=self.encoder.config.hidden_size)

        # 词性特征
        self.use_pos_tag = use_pos_tag
        self.use_lstm = use_lstm
        
        if self.use_pos_tag:
            self.pos_tag_to_idx = pos_tag_to_idx
            self.pos_tag_coprus_size = pos_tag_coprus_size
            self.pos_tag_pad_idx = pos_tag_pad_idx
            self.pos_tag_embed_dim = pos_tag_embed_dim if pos_tag_embed_dim is not None else self.encoder.config.hidden_size
            # 词性编码
            self.pos_tag_embeddings = nn.Embedding(num_embeddings=pos_tag_coprus_size,
                                                   embedding_dim=self.pos_tag_embed_dim,
                                                   padding_idx=pos_tag_pad_idx
                                                )
            if self.use_lstm:
                self.bilstm = nn.LSTM(
                    input_size=self.encoder.config.hidden_size,
                    hidden_size=self.encoder.config.hidden_size,
                    num_layers=1,
                    dropout=dropout_rate,
                    batch_first=True,
                    bidirectional=True
                )
                # 分类预测头
                self.feedforward = nn.Linear(in_features=2*self.encoder.config.hidden_size + self.pos_tag_embed_dim, out_features=self.target_size)
            else:
                # 分类预测头
                self.feedforward = nn.Linear(in_features=self.encoder.config.hidden_size + self.pos_tag_embed_dim, out_features=self.target_size)
        else:
            if self.use_lstm:
                self.bilstm = nn.LSTM(
                    input_size=self.encoder.config.hidden_size,
                    hidden_size=self.encoder.config.hidden_size,
                    num_layers=1,
                    dropout=dropout_rate,
                    batch_first=True,
                    bidirectional=True
                )
                self.feedforward = nn.Linear(in_features=2*self.encoder.config.hidden_size, out_features=self.target_size)
            else:
                # 分类预测头
                self.feedforward = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=self.target_size)

        # 使用CRF层
        self.use_crf = use_crf
        if self.use_crf:
            self.crf_layer = ConditionalRandomField(num_tags=self.target_size, constraints=allowed_transitions(constraint_type="BIO", labels=self.id_to_tag))
        
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
            self.loss_weight = torch.tensor(self.loss_weight, device=self.gpu)
            
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
        self.config = {k: v for k, v in locals().items() if k not in ["self", "__class__", "argv", "gpu"]}
        self.config = PretrainedConfig.from_dict(self.config)
        self.gpu = gpu
    
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
        print("debug from_pretrained: model's device = ", next(model.parameters()).device)
        model = model.to(model.gpu)
        model_path = get_pretrained_model_path(model_dir)
        model = load_wise_model(model, model_path, gpu=model.gpu)
        
        return model

    def save_pretrained(self, model_dir, filename):
        out_model_path = os.path.join(model_dir, filename)
        save_wise_model(self, out_model_path)
        self.save_config(model_dir)

    def forward(self, batch, mode='train', use_post=True, post_method=[1,2], verbose=False):
        tokens, tags, mask, token_mask, metadata, pos_tag_idx, rule_based_tags_rep, sentence_str = batch
        tokens, tags, mask, token_mask = tokens.to(self.gpu), tags.to(self.gpu), mask.to(self.gpu), token_mask.to(self.gpu)

        tokens_lens = torch.sum(mask, dim=-1) # (B, l) -> (B)

        if pos_tag_idx is not None:
            pos_tag_idx = pos_tag_idx.to(self.gpu)
        batch_size = tokens.size(0)

        embedded_text_input = self.encoder(input_ids=tokens, attention_mask=mask.float().to(self.gpu))
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        if self.use_position:
            embedded_text_input =  self.position_embedding(embedded_text_input)

        if self.use_pos_tag:
            embedded_pos_tag = self.pos_tag_embeddings(pos_tag_idx)
            embedded_pos_tag_ = embedded_pos_tag * mask.unsqueeze(2)
            embedded_text_input = torch.cat((embedded_text_input, embedded_pos_tag_), dim=-1)
        
        if self.use_lstm:
            tokens_lens = tokens_lens.cpu()
            embedded_text_pack = pack_padded_sequence(embedded_text_input, tokens_lens, batch_first=True, enforce_sorted=False)
            embedded_text_input, _ = self.bilstm(embedded_text_pack)
            embedded_text_input, _ = pad_packed_sequence(embedded_text_input, batch_first=True)
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))
            
        token_scores = self.feedforward(embedded_text_input)

        # project the token representation for classification
        token_log_scores = F.log_softmax(token_scores, dim=-1)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(tokens=tokens, token_scores=token_scores, token_log_scores=token_log_scores, mask=mask, tags=tags, metadata=metadata, rule_based_tags_rep=rule_based_tags_rep, batch_size=batch_size, mode=mode, use_post=use_post, post_method=post_method, verbose=verbose)
        return output


    def _compute_token_tags(self, tokens, token_scores, token_log_scores, mask, tags, metadata, rule_based_tags_rep, batch_size, mode='train', use_post=True, post_method=[1,2], verbose=False):
        # compute the log-likelihood loss and compute the best NER annotation sequence
        if self.use_crf:
            loss = -self.crf_layer(token_log_scores, tags, mask) / float(batch_size)
            best_path = self.crf_layer.viterbi_tags(token_log_scores, mask)
        else:
            assert self.use_cls is not None
            flat_loss = self.criterion(
                token_log_scores.view(-1, len(self.id_to_tag)), tags.view(-1))
            if self.use_wof:
                masks_of_entity = tags.view(-1) != self.tag_to_id["O"]
                weights_of_loss = masks_of_entity.float().to(self.gpu) + 0.5 # B  *S
                loss = torch.sum(flat_loss * weights_of_loss) / float(batch_size)
            else:
                loss = torch.sum(flat_loss) / float(batch_size)

            best_path = torch.argmax(token_log_scores, dim=-1).cpu().detach().numpy().tolist()

        seq_len = tokens.shape[1]
        pred_results, pred_tags, pred_tag_idxs = [], [], []
        for i in range(batch_size):
            if self.use_crf:
                tag_seq, _ = best_path[i]
            else:
                assert self.use_cls is not None
                tag_seq = best_path[i]

            tagnames = [self.id_to_tag[x] for x in tag_seq]
            if mode != "train" and self.tongxiao_post:
                rule_tags = rule_based_tags_rep[i]
                tagnames = [tagnames[idx] if not rule_tags[idx] else rule_tags[idx] for idx in range(len(tagnames))]

            _tag_idxs = [self.tag_to_id[t] for t in tagnames] + [self.tag_to_id['O'] ]*(seq_len - len(tagnames))
            pred_tags.append([ self.id_to_tag[x] for x in _tag_idxs ])
            pred_tag_idxs.append( _tag_idxs )
            pred_results.append(extract_spans(tagnames, iob_tagging=self.iob_tagging))

        # 后处理
        if use_post and mode != "train":
            pred_results = post_process(
                        self.tokenizer,
                        pred_results, 
                        tokens.cpu().detach().numpy().tolist(), 
                        pred_tags,
                        post_method=post_method,
                        mode=mode,
                        verbose=verbose
                        )

        self.span_f1(pred_results, metadata)
        output = {"loss": loss, "results": self.span_f1.get_metric()}

        if mode in ['predict', 'ensemble']:
            output["predictions"] = {
                'token_tags': pred_tags,
                'pred_spans': pred_results,
                'gold_spans': metadata,
            }
        if mode == 'ensemble':
            logits = F.softmax(token_scores)
            
            batch_size, seq_len, cls_num = logits.shape # class
            
            active_cls_idx = torch.tensor(pred_tag_idxs).to(self.gpu).unsqueeze(2).expand(-1, -1, cls_num) # (B, s, c)
            
            active_mask = torch.arange(0, cls_num, device=self.gpu).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, cls_num) == active_cls_idx

            logits = logits * active_mask

            logits = logits.cpu().detach()
            
            output["predictions"]["logits"] = logits

        return output

    def predict_tags(self, batch, use_post=True):
        tokens, tags, mask, token_mask, metadata, pos_tag_idx, rule_based_tags, sentence_str = batch
        
        pred_tags = self.forward(batch, mode='predict', use_post=use_post)['predictions']['token_tags']
        tag_results = [compress(pred_tags_, mask_) for pred_tags_, mask_ in zip(pred_tags, token_mask)]
        return tag_results