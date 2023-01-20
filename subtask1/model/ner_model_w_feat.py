from pickle import NONE
from typing import List, Any
from itertools import compress

import pytorch_lightning.core.lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import math


import torch
import torch.nn.functional as F
import numpy as np

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModel

from log import logger
from utils.metric import SpanF1
from utils.reader_utils import extract_spans, get_tags


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
 
    def forward(self, output, target):
        # output: (batch*len, c) , target:(batch*len)
        # convert output to presudo probability
        out_target = torch.stack([output[i, t] for i, t in enumerate(target)])
        probs = torch.sigmoid(out_target)
        focal_weight = torch.pow(1-probs, self.gamma)
 
        # add focal weight to cross entropy
        ce_loss = F.cross_entropy(output, target, weight=self.weight, reduction='none')
        focal_loss = focal_weight * ce_loss
 
        if self.reduction == 'mean':
            focal_loss = (focal_loss/focal_weight.sum()).sum()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError("Illegal Loss!")
        return focal_loss

class PositionalEncoding(nn.Module):
  
    def __init__(self, d_model, max_len=256):
        """
        d_model: embedding dim
        max_len：这里指的是 句子最大数量
        """
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #.transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        # return x + self.pe[:x.size(0), :]
        return x + self.pe[:, :x.size(1)]


class NERBaseAnnotator_W_FEAT(pl.LightningModule):
    def __init__(self,
                 train_data=None,
                 dev_data=None,
                 lr=1e-5,
                 dropout_rate=0.1,
                 batch_size=16,
                 tag_to_id=None,
                 stage='fit',
                 
                 iob_tagging="conll",

                 pad_token_id=1,
                 encoder_model='xlm-roberta-base',
                 num_gpus=1,

                 use_pos_tag=False, 
                 pos_tag_to_idx={},
                 pos_tag_coprus_size=None,
                 pos_tag_embed_dim=None,
                 pos_tag_pad_idx=0,

                 use_position=False,

                 use_crf=True,
                 use_cls=None,
                 use_wof=False,
                 ):
        super(NERBaseAnnotator_W_FEAT, self).__init__()

        self.train_data = train_data
        self.dev_data = dev_data

        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id
        self.batch_size = batch_size

        self.stage = stage
        self.iob_tagging = iob_tagging
        self.num_gpus = num_gpus
        self.target_size = len(self.id_to_tag)

        # set the default baseline model here
        self.pad_token_id = pad_token_id

        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict=True)

        # 位置编码
        self.use_position = use_position
        if self.use_position:
            self.position_embedding = PositionalEncoding(d_model=self.encoder.config.hidden_size)

        # 词性特征
        self.use_pos_tag = use_pos_tag
        if self.use_pos_tag:
            self.pos_tag_to_idx = pos_tag_to_idx
            self.pos_tag_coprus_size = pos_tag_coprus_size
            self.pos_tag_pad_idx = pos_tag_pad_idx
            self.pos_tag_embed_dim = pos_tag_embed_dim if pos_tag_embed_dim is not None else self.encoder.config.hidden_size
            self.pos_tag_embeddings = nn.Embedding(num_embeddings=pos_tag_coprus_size,
                                                   embedding_dim=self.pos_tag_embed_dim,
                                                   padding_idx=pos_tag_pad_idx
                                                )
            # 分类预测头
            self.feedforward = nn.Linear(in_features=self.encoder.config.hidden_size + self.pos_tag_embed_dim, out_features=self.target_size)
        else:
            # 分类预测头
            self.feedforward = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=self.target_size)

        # CRF层
        self.use_crf = use_crf
        if self.use_crf:
            self.crf_layer = ConditionalRandomField(num_tags=self.target_size, constraints=allowed_transitions(constraint_type="BIO", labels=self.id_to_tag))
        
        # 分类层-损失
        self.use_cls = use_cls
        self.use_wof = use_wof
        if use_cls is not None:
            assert use_cls in ["crossentropy", "focal"]
            self.loss_weight = [1.0]*len(self.id_to_tag)
            self.loss_weight[ self.tag_to_id["B-OBJ_NAME"] ] = 5.0
            self.loss_weight[ self.tag_to_id["I-OBJ_NAME"] ] = 5.0
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

        self.lr = lr
        self.dropout = nn.Dropout(dropout_rate)

        # SpanF1 is required to evaluate the model - don't remove
        self.span_f1 = SpanF1()
        self.setup_model(self.stage)
        self.save_hyperparameters('pad_token_id', 'encoder_model',
                                    'use_pos_tag', 'pos_tag_coprus_size',
                                    'pos_tag_embed_dim', 'pos_tag_pad_idx',
                                    'pos_tag_to_idx',

                                    'use_position',
                                    'use_crf',
                                    'use_cls',
                                    'use_wof'
                                    )

    def setup_model(self, stage_name):
        if stage_name == 'fit' and self.train_data is not None:
            # Calculate total steps
            train_batches = len(self.train_data) // (self.batch_size * self.num_gpus)
            self.total_steps = 50 * train_batches # old: 50 25

            self.warmup_steps = int(train_batches * 0.5)

    def collate_batch(self, batch):
        if len(batch[0]) != 7:
            print(batch[0])
            raise ValueError("what the fuck1")
        else:
            # print("pass1")
            pass

        batch_ = list(zip(*batch))
        if len(batch_) != 7:
            print(batch_)
            raise ValueError("what the fuck2")
        else:
            # print("pass2")
            pass
        tokens, masks, token_masks, gold_spans, tags, poss, sentence_str = batch_[0], batch_[1], batch_[2], batch_[3], batch_[4], batch_[5], batch_[6]

        max_len = max([len(token) for token in tokens])
        token_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.pad_token_id)
        tag_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.tag_to_id['O'])
        
        if self.use_pos_tag:
            pos_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.pos_tag_to_idx['PAD'])
        
        mask_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
        token_masks_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)

        for i in range(len(tokens)):
            tokens_ = tokens[i]
            seq_len = len(tokens_)

            token_tensor[i, :seq_len] = tokens_
            tag_tensor[i, :seq_len] = tags[i]

            if self.use_pos_tag:
                pos_tensor[i, :seq_len] = poss[i]

            mask_tensor[i, :seq_len] = masks[i]
            token_masks_tensor[i, :seq_len] = token_masks[i]

        if self.use_pos_tag:
            return token_tensor, tag_tensor, mask_tensor, token_masks_tensor, gold_spans, pos_tensor, sentence_str
        else:
            return token_tensor, tag_tensor, mask_tensor, token_masks_tensor, gold_spans, None, sentence_str

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        if self.stage == 'fit':
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.1, verbose=True)
            # lr_scheduler={
            #     'scheduler': scheduler,
            #     'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
            #     'interval': 'epoch',
            #     'frequency': 1, # The frequency of the scheduler
            #     'monitor': 'val_micro@F1', #监听数据变化
            #     'name':"lr_scheduler",
            # }
            # return {
            #     'optimizer': optimizer,
            #     'lr_scheduler': lr_scheduler,
            # }
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps)
            lr_scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [lr_scheduler]

        return [optimizer]

    def train_dataloader(self):
        loader = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=12)
        return loader

    def val_dataloader(self):
        if self.dev_data is None:
            return None
        loader = DataLoader(self.dev_data, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=12)
        return loader

    def training_epoch_end(self, outputs: List[Any]) -> None:
        pred_results = self.span_f1.get_metric(True)
        avg_loss = np.mean([preds['loss'].item() for preds in outputs])
        self.log_metrics(pred_results, loss=avg_loss, suffix='', on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch)
        self.log_metrics(output['results'], loss=output['loss'].item(), suffix='', on_step=True, on_epoch=False)
        
        return output
    
    def validation_epoch_end(self, outputs: List[Any]) -> None:
        pred_results = self.span_f1.get_metric(True)
        avg_loss = np.mean([preds['loss'].item() for preds in outputs])
        self.log_metrics(pred_results, loss=avg_loss, suffix='val_', on_step=False, on_epoch=True)
        
        print("="*100)
        print("="*100)
        for key in ["F1@CONST_DIR", "F1@LIMIT", "F1@OBJ_DIR", "F1@OBJ_NAME", "F1@PARAM", "F1@VAR", "micro@F1"]:
            _key =  f"val_{key}"
            print(f"{_key}:   {pred_results[key]}")
        print("="*100)
        print("="*100)

    def validation_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch)
        self.log_metrics(output['results'], loss=output['loss'].item(), suffix='val_', on_step=True, on_epoch=False)
        return output

    def test_epoch_end(self, outputs):
        pred_results = self.span_f1.get_metric()
        avg_loss = np.mean([preds['loss'].item() for preds in outputs])
        self.log_metrics(pred_results, loss=avg_loss, on_step=False, on_epoch=True)

        out = {"test_loss": avg_loss, "results": pred_results}
        return out

    def test_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch, mode=self.stage)
        self.log_metrics(output['results'], loss=output['loss'].item(), suffix='_t', on_step=True, on_epoch=False)
        
        """Important !!! Make batch_size = 1 if log debug info"""
        print(f"############# [batch_idx {batch_idx}] #############")
        print("sentence_str : ", batch[-1])
        tok_list, tok_tag_list = batch[-1][0].split(), batch[1][0].cpu().detach().numpy().tolist()[1:-1]
        print("token_tags : ", [(i, tok, tag) for i, (tok, tag) in enumerate(zip(tok_list, [self.id_to_tag[x] for x in tok_tag_list]))] )
        print()

        gold_spans = [(" ".join(tok_list[k[0]-1: k[1]+1-1]), k, v) for k, v in output["gold_spans"][0].items()]
        pred_spans = [(" ".join(tok_list[k[0]-1: k[1]+1-1]), k, v) for k, v in output["pred_spans"][0].items()]
        print("gold_spans : ", gold_spans)
        print("pred_spans : ", pred_spans)
        print()
        gold_spans_set = set(gold_spans)
        pred_spans_set = set(pred_spans)
        diff_1  = list(pred_spans_set - gold_spans_set)
        diff_1 = sorted(diff_1,key=lambda x: x[1][0])
        print(f"pred - gold ( {len(diff_1)} ) : {diff_1}")
        print()
        diff_2 = list(gold_spans_set - pred_spans_set)
        diff_2 = sorted(diff_2,key=lambda x: x[1][0])
        print(f"gold - pred ( {len(diff_2)} ) : {diff_2}")
        print("="*100)
        
        return output

    def log_metrics(self, pred_results, loss=0.0, suffix='', on_step=False, on_epoch=True):
        for key in pred_results:
            self.log(suffix + key, pred_results[key], on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

        self.log(suffix + 'loss', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

    def perform_forward_step(self, batch, mode=''):
        
        tokens, tags, mask, token_mask, metadata, pos_tag_idx, sentence_str = batch
        batch_size = tokens.size(0)

        embedded_text_input = self.encoder(input_ids=tokens, attention_mask=mask.float())
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        if self.use_position:
            embedded_text_input =  self.position_embedding(embedded_text_input)

        if self.use_pos_tag:
            embedded_pos_tag = self.pos_tag_embeddings(pos_tag_idx)
            embedded_pos_tag_ = embedded_pos_tag * mask.unsqueeze(2)
            embeded_enhanced = torch.cat((embedded_text_input, embedded_pos_tag_), dim=-1)
            token_scores = self.feedforward(embeded_enhanced)
        else:
            token_scores = self.feedforward(embedded_text_input)

        # project the token representation for classification
        token_scores = F.log_softmax(token_scores, dim=-1)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(token_scores=token_scores, mask=mask, tags=tags, metadata=metadata, batch_size=batch_size, mode=mode)
        return output

    def _compute_token_tags(self, token_scores, mask, tags, metadata, batch_size, mode=''):
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

        pred_results, pred_tags = [], []
        for i in range(batch_size):
            if self.use_crf:
                tag_seq, _ = best_path[i]
            else:
                assert self.use_cls is not None
                tag_seq = best_path[i]
            
            pred_tags.append([self.id_to_tag[x] for x in tag_seq])
            pred_results.append(extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag], iob_tagging=self.iob_tagging))

        self.span_f1(pred_results, metadata)
        output = {"loss": loss, "results": self.span_f1.get_metric()}

        if mode == 'predict':
            output['token_tags'] = pred_tags
            output['pred_spans'] = pred_results
            output['gold_spans'] = metadata
        return output

    def predict_tags(self, batch, device='cuda:0'):
        tokens, tags, mask, token_mask, metadata = batch
        tokens, mask, token_mask, tags = tokens.to(device), mask.to(device), token_mask.to(device), tags.to(device)
        batch = tokens, tags, mask, token_mask, metadata

        pred_tags = self.perform_forward_step(batch, mode='predict')['token_tags']
        tag_results = [compress(pred_tags_, mask_) for pred_tags_, mask_ in zip(pred_tags, token_mask)]
        return tag_results
