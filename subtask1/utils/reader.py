import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
import random
from log import logger
from nltk.stem import PorterStemmer
from utils.reader_utils import get_ner_reader, extract_spans, _assign_ner_tags, get_border_vocab_from_tag_vocab, _assign_ner_tags_for_rule_tags
import re
import os
import spacy
nlp = spacy.load("en_core_web_sm")
import string
os.environ["TOKENIZERS_PARALLELISM"] = "false"

times_idx = 0
total_number_idx = 0
_known = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90,

    'hundred': 100,
    'thousand': 1000,
    'million': 1000000,

    "first": 1,
    "second": 2,
    "third": 3,
    "fifth": 5,
    "eighth": 8,
    "ninth": 9,
    "twelfth": 12,

    "once": 1,
    "twice": 2,
    
    "half": 0.5,
}

def spoken_word_to_number(token):
    """
    更复杂的实现参考
     (1): https://www.jb51.net/article/68826.htm
     (2): https://blog.csdn.net/weixin_44454792/article/details/125219692
    """
    token = token.lower().strip()
    if token in _known:
        # 处理一般 数字单词
        return str(_known[token])
    elif token[:-3] in _known:
        # 处理 sixth 这样的序数词
        return str(_known[token[:-3]])
    else:
        # 其他非数字
        return token


def generage_rule_based_tags(tokens_: list, tongxiao_post: bool = False):
    '''
    产生基于规则的 tags
    '''
    # 产生基于规则的 tags
    rule_based_tags = [None] * len(tokens_)
    if not tongxiao_post:
        return rule_based_tags
    stemmer = PorterStemmer()
    # 词 'total' 相关的规则
    total_idices = [idx for idx, token in enumerate(tokens_) if stemmer.stem(token.lower()) == 'total']
    for idx in total_idices:
        next_token = stemmer.stem(tokens_[idx + 1].lower())
        prev_token = stemmer.stem(tokens_[idx - 1].lower())
        if next_token == "number":
            global total_number_idx

            """ TO TEST """
            rule_based_tags[idx] = 'O'
            rule_based_tags[idx + 1] = 'B-OBJ_NAME'
            # if total_number_idx < 55:
            #     rule_based_tags[idx] = 'O'
            #     rule_based_tags[idx + 1] = 'B-OBJ_NAME'
            # else:
            #     rule_based_tags[idx] = 'B-OBJ_NAME'
            #     rule_based_tags[idx + 1] = 'I-OBJ_NAME'
            
            
            if tokens_[idx - 1].lower() == "the":
                # 将 the 赋为 O
                rule_based_tags[idx - 1] = 'O'
            if tokens_[idx + 2].lower() == "of":
                # 将 of 赋为 I-OBJ_NAME
                rule_based_tags[idx + 2] = 'I-OBJ_NAME'
            total_number_idx += 1
        elif next_token == "amount":
            # 将 total number & total amount 赋值为 O, B-OBJ_NAME
            rule_based_tags[idx] = 'O'
            rule_based_tags[idx + 1] = 'B-OBJ_NAME'
            if stemmer.stem(tokens_[idx - 1].lower()) == "the":
                # 将 the 赋为 O
                rule_based_tags[idx - 1] = 'O'
            if stemmer.stem(tokens_[idx + 2].lower()) == "of":
                # 将 of 赋为 I-OBJ_NAME
                rule_based_tags[idx + 2] = 'I-OBJ_NAME'
        elif next_token in ["time", "radiat", "unit", "product"]:
            # 将 total time, total radiation, total units, total production 赋值为 B-OBJ_NAME, I-OBJ_NAME
            # 由于 stemmer 的存在, radiation, units, production 会被还原为 radiat, unit, product
            rule_based_tags[idx] = 'B-OBJ_NAME'
            rule_based_tags[idx + 1] = 'I-OBJ_NAME'
            if stemmer.stem(prev_token.lower()) == "the":
                # 将 the 赋为 O
                rule_based_tags[idx - 1] = 'O'
            if stemmer.stem(tokens_[idx + 2].lower()) == "of":
                # 将 of 赋为 I-OBJ_NAME
                rule_based_tags[idx + 2] = 'I-OBJ_NAME'
        # elif next_token == 'production':
        #     # 将 total production 赋值为 B-OBJ_NAME, I-OBJ_NAME
        #     rule_based_tags[idx] = 'B-OBJ_NAME'
        #     rule_based_tags[idx + 1] = 'I-OBJ_NAME'
        elif prev_token == 'a':
            # 将 a total 赋值为 O, O
            rule_based_tags[idx - 1] = 'O'
            rule_based_tags[idx] = 'O'
            if next_token == 'of':
                rule_based_tags[idx + 1] = 'O'
        else:
            # TO TEST
            rule_based_tags[idx] = 'O'
    
    # 词 'time' 相关的规则
    time_idices = [idx for idx, token in enumerate(tokens_) if stemmer.stem(token.lower()) == 'time']
    for idx in time_idices:
        prev_token = stemmer.stem(tokens_[idx - 1].lower())
        if prev_token in _known.keys() or prev_token.replace(',', '').replace('.', '').replace('%', '').isnumeric():
            # prev_token 为数字, 赋为 B-PARAM, O (还有 B-LIMIT, O 需考虑)
            rule_based_tags[idx - 1] = 'B-PARAM'
            global times_idx
            """ TO TEST """
            # rule_based_tags[idx] = 'O' if times_idx < 25 else 'I-PARAM'
            rule_based_tags[idx] = 'O'

            times_idx += 1
        elif prev_token == 'a':
            # 将 a time 赋为 O, O
            rule_based_tags[idx - 1] = 'O'
            rule_based_tags[idx] = 'O'
    
    # # 词 'amount' 相关的规则
    # amount_idices = [idx for idx, token in enumerate(tokens_) if stemmer.stem(token) == 'amount']
    # for idx in amount_idices:
    #     rule_based_tags[idx] = 'B-OBJ_NAME'
    # 词 'hour' 相关的规则
    hour_idices = [idx for idx, token in enumerate(tokens_) if stemmer.stem(token) == 'hour']
    for idx in hour_idices:
        prev_token = stemmer.stem(tokens_[idx - 1])
        if prev_token == 'per':
            # 将 per hour 赋为 O, O
            rule_based_tags[idx - 1] = 'O'
            rule_based_tags[idx] = 'O'

        """ TO TEST """
        # elif prev_token in _known.keys() or prev_token.replace(',', '').replace('.', '').replace('%', '').isnumeric():
        #     # prev_token 为数字, 赋为 B-PARAM, O
        #     rule_based_tags[idx - 1] = 'B-PARAM'
        #     rule_based_tags[idx] = 'O'
    # 词 'minutes' 相关的规则
    minutes_idices = [idx for idx, token in enumerate(tokens_) if token == 'minutes']
    for idx in minutes_idices:
        next_token, next_next_token = tokens_[idx + 1], tokens_[idx + 2]
        if next_token == 'to':
            if next_next_token in ['be', 'make']:
                rule_based_tags[idx] = 'B-OBJ_NAME'
            elif next_next_token in ['cook', 'bake']:
                rule_based_tags[idx] = 'B-OBJ_NAME'
                rule_based_tags[idx + 1] = 'I-OBJ_NAME'
                rule_based_tags[idx + 2] = 'I-OBJ_NAME'
        elif next_token == '.':
            rule_based_tags[idx] == 'B-OBJ_NAME'
        else:
            rule_based_tags[idx] == 'O'
    return rule_based_tags


class CoNLLReader(Dataset):
    def __init__(self, max_instances=-1, max_length=50, target_vocab=None, pretrained_dir='', encoder_model='xlm-roberta-base',
                 
                 use_num_standardize=False,
                 use_punc_standardize=True,
                 use_label_standardize=True,
                 label_method=[0],
                 
                 use_pos_tag=False,
                 tongxiao_post=False,
                 use_lemma=False,
                 pos_tag_to_idx={},
                 
                 iob_tagging="conll",
                 tag_to_id={},
                 mode="train",

                 use_dynamic_mask=False,

                 **argv
                 ):
        self._max_instances = max_instances
        self._max_length = max_length
        print("[DEBUG] CoNLLReader", pretrained_dir + encoder_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir + encoder_model)

        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']

        self.label_to_id = {} if target_vocab is None else target_vocab
        self.instances = []

        self.use_num_standardize = use_num_standardize
        self.use_punc_standardize = use_punc_standardize
        self.use_label_standardize = use_label_standardize
        self.label_method = label_method
        print(f"DEBUg CoNLLReader: use_label_standardize = {use_label_standardize}, label_method={label_method}")
        self.use_pos_tag = use_pos_tag
        self.tongxiao_post = tongxiao_post
        self.use_lemma = use_lemma
        self.pos_tag_to_idx = pos_tag_to_idx


        self.use_dynamic_mask = use_dynamic_mask

        self.mode = mode
        self.iob_tagging = iob_tagging
        self.tag_to_id = tag_to_id

    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data):
        dataset_name = data if isinstance(data, str) else 'dataframe'
        logger.info('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, metadata in get_ner_reader(data=data):
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, coded_pos_, gold_spans_, mask, rule_based_tags_rep = self.parse_line_for_ner(fields=fields)

            tokens_tensor = torch.tensor(tokens_sub_rep, dtype=torch.long)
            tag_tensor = torch.tensor(coded_ner_, dtype=torch.long)

            if self.use_pos_tag:
                pos_tensor = torch.tensor(coded_pos_, dtype=torch.long)
            else:
                pos_tensor = None

            token_masks_rep = torch.tensor(token_masks_rep)
            mask_rep = torch.tensor(mask)

            self.instances.append((tokens_tensor, mask_rep, token_masks_rep, gold_spans_, tag_tensor, pos_tensor, rule_based_tags_rep, sentence_str))
            # self.instances.append((tokens_tensor, mask_rep, token_masks_rep, gold_spans_, tag_tensor))
            instance_idx += 1
        logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def parse_line_for_ner(self, fields):
        rule_based_tags = generage_rule_based_tags(fields[0], self.tongxiao_post)
        if self.mode == "train":
            # 动态MASK
            if self.use_dynamic_mask:
                for i, ner_tag in enumerate(fields[3]):
                    if ner_tag=="O" and random.random() < 0.15:
                        fields[0][i] = self.tokenizer.mask_token
        elif self.mode == "valid":
            pass
        else:
            assert self.mode == "test"
        # 预处理获取特征
        if self.use_lemma is True or self.use_pos_tag is True:
            fields = self.preprocess(fields)

        if self.use_lemma is True:
            tokens_, ner_tags = fields[1], fields[3]
        else:
            tokens_, ner_tags = fields[0], fields[3]
        
        if self.use_pos_tag is True:
            pos_tags = fields[2]
        else:
            pos_tags = None

        sentence_str, tokens_sub_rep, ner_tags_rep, pos_tags_rep, token_masks_rep, mask, rule_based_tags_rep = self.parse_tokens_for_ner(tokens_, ner_tags, rule_based_tags, pos_tags)
        # subwords, subword_idxs, subword_labels, subword_mask, subword_pos, mask
        gold_spans_ = extract_spans(ner_tags_rep, iob_tagging=self.iob_tagging)
        coded_ner_ = [self.label_to_id[tag] for tag in ner_tags_rep]
        if self.use_pos_tag:
            coded_pos_ = [self.pos_tag_to_idx[pos] if pos in self.pos_tag_to_idx else self.pos_tag_to_idx['UNK'] for pos in pos_tags_rep]
        else:
            coded_pos_ = None
        return sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, coded_pos_, gold_spans_, mask, rule_based_tags_rep

    def preprocess_for_spacy(self, fields):
        _tokens = " ".join(fields[0])
        
        detail_tokens = [token for token in nlp(_tokens) if token.text]
        raw_token = [t for t in fields[0]]

        if len(raw_token) != len(detail_tokens):
            print(_tokens)
            print()
            print([t.text for t in detail_tokens])
            raise ValueError("preprocess Error!")

        for i, detail_token in enumerate(detail_tokens):
            fields[1][i] = detail_token.lemma_
            fields[2][i] = detail_token.pos_
        
        return fields

    def number_standardize(self, token, idx, tokens_):
        """将部分英文数字(如two) 转化为 数字(如2)"""
        token = spoken_word_to_number(token)

        """将特殊数字1,600标准化为1600"""
        if re.match(r"\d+,\d+", token):
            token = token.replace(",", "")

        # 处理 a (a third)
        if token == "a" and idx+1 < len(tokens_) and tokens_[idx+1] in _known:
            token = "1"

        # 处理小数
        if "." in token and re.findall(r"\d", token):
            # 为小数拼接特殊符号num
            # "在 roberta 中， 一些 小数如0.9 可能分错为 _ 0.9, 这里的_干扰了判断
            token = f"num{token}"

        return token
    
    def label_standardize(self, token, ner_tag, idx, tokens_, ner_tags):
        
        if 0 in self.label_method:
            # # 规范化数字标签 num times 标签
            if token in ["times"] and ner_tag in ["B-LIMIT", "I-LIMIT", "B-PARAM", "I-PARAM"]:
                ner_tag = "O"
        
        if 1 in self.label_method:
            # 规范化数字标签 num% 和 num times
            # NV 1 (add %)
            if token in ["%"] and ner_tag == "O" and ner_tags[idx-1] in ["B-LIMIT", "I-LIMIT", "B-PARAM", "I-PARAM"]:
                if self.iob_tagging not in ["conll_ob"]:
                    ner_tag = ner_tags[idx-1].replace("B-", "I-")
                else:
                    ner_tag = ner_tags[idx-1]
        
        # NV 2 (add times and %)
        if 2 in self.label_method:
            if token in ["%", "times"] and ner_tag == "O" and ner_tags[idx-1] in ["B-LIMIT", "I-LIMIT", "B-PARAM", "I-PARAM"]:
                if self.iob_tagging not in ["conll_ob"]:
                    ner_tag = ner_tags[idx-1].replace("B-", "I-")
                else:
                    ner_tag = ner_tags[idx-1]

        # NV 3
        if 3 in self.label_method:
            # 3.1 规范化OBJ_NAME标签 (total) number of obj 和 (total) obj 标注（使用前者）
            if token=="total" and idx+1 < len(tokens_) and ner_tags[idx+1] in ["B-OBJ_NAME"]:
                ner_tag = "B-OBJ_NAME"
            if ner_tag == "B-OBJ_NAME" and idx-1 >= 0 and tokens_[idx-1] == "total":
                ner_tag = "I-OBJ_NAME"
            
            # 3.2 规范化OBJ_NAME标签 (the) total number of obj (去掉the)
            if token=="the" and ner_tag == "B-OBJ_NAME":
                ner_tag = "O"
        
        # NV 4
        if 4 in self.label_method:
            # 规范化VAR标签： 如 and 等词的情况，VAR1 and VAR2, VAR and O
            if token=="and" and ner_tags[idx+1] in ["B-VAR", "O"]: # V4.2 VAR and O
                ner_tag = "O"
        
        return ner_tag

    def punc_standardize(self, sentence_str, ner_tags_rep, pos_tags_rep):
        # V old
        # if sentence_str.split()[-1] in string.punctuation:
        #     ner_tags_rep[-1] = "O"
        #     if self.use_pos_tag:
        #         pos_tags_rep[-1] = "PUNCT"

        # V new
        if sentence_str.split()[-1] in string.punctuation:
            # 处理 xxx.-> _xxx . 和 .-> _ . 的分词情况
            ner_tags_rep[-1] = "O"
            if self.use_pos_tag:
                pos_tags_rep[-1] = "PUNCT"
            # 处理 .->_ . 的分词情况
            if sentence_str.split()[-2] == '▁':
                ner_tags_rep[-2] = "O"
                if self.use_pos_tag:
                    pos_tags_rep[-2] = "PUNCT"
        return ner_tags_rep, pos_tags_rep

    def parse_tokens_for_ner(self, tokens_, ner_tags, rule_based_tags, pos_tags=None):
        sentence_str = ''
        # 增加开始标志
        tokens_sub_rep, ner_tags_rep = [self.pad_token_id], ['O']
        rule_based_tags_rep = [None]
        token_masks_rep = [False]
        pos_tags_rep = ["UNK"] # 0 代表 UNK_POS
        for idx, token in enumerate(tokens_):
            if self._max_length != -1 and len(tokens_sub_rep) > self._max_length:
                break

            ner_tag = ner_tags[idx]

            """1. 数字标准化"""
            if self.use_num_standardize:
                token = self.number_standardize(token.lower(), idx, tokens_)
            else:
                token = token.lower()
            
            sub_tokens = self.tokenizer.tokenize(token)
            sentence_str += ' ' + ' '.join(sub_tokens)
            sentence_str_tokens = sentence_str.split()
            rep_ = self.tokenizer(token)['input_ids']
            rep_ = rep_[1:-1]
            tokens_sub_rep.extend(rep_)
            
            """ 2. 标签规范化 """
            # if self.mode in ["train", "dev", "test"] and self.use_label_standardize:
            # if self.mode in ["train", "dev"] and self.use_label_standardize:
            # if self.mode in ["test"] and self.use_label_standardize:
            # if False:
            if self.use_label_standardize:
                ner_tag = self.label_standardize(token, ner_tag, idx, tokens_, ner_tags)
            
            # 测试：
            # if token in ["times"] and ner_tag in ["B-LIMIT", "I-LIMIT", "B-PARAM", "I-PARAM"]:
            #     ner_tag = "O"
            
            """ 3. 处理子词 """
            # if we have a NER here, in the case of B, the first NER tag is the B tag, the rest are I tags.
            tags, masks = _assign_ner_tags(ner_tag, rep_, iob_tagging=self.iob_tagging)
            rule_tags, _ = _assign_ner_tags_for_rule_tags(rule_based_tags[idx], rep_)

            ner_tags_rep.extend(tags)
            token_masks_rep.extend(masks)
            rule_based_tags_rep.extend(rule_tags)
            
            if self.use_pos_tag:
                pos_tag = pos_tags[idx]
                _tags = [pos_tag] * len(tags)
                pos_tags_rep.extend(_tags)
    
            """ 4. 标点符号更正 """
            if self.use_punc_standardize:
                ner_tags_rep, pos_tags_rep = self.punc_standardize(sentence_str, ner_tags_rep, pos_tags_rep)

        # 增加末尾标志
        tokens_sub_rep.append(self.pad_token_id)
        rule_based_tags_rep.append(None)

        ner_tags_rep.append('O')
        pos_tags_rep.append('PAD')

        token_masks_rep.append(False)
        mask = [True] * len(tokens_sub_rep)
        
        return sentence_str, tokens_sub_rep, ner_tags_rep, pos_tags_rep, token_masks_rep, mask, rule_based_tags_rep

    def collate_batch_fn(self, batch):
        if len(batch[0]) != 8:
            print(batch[0])
            raise ValueError("what the fuck1")
        else:
            pass
        batch_ = list(zip(*batch))
        if len(batch_) != 8:
            print(batch_)
            raise ValueError("what the fuck2")
        else:
            pass
        tokens, masks, token_masks, gold_spans, tags, poss, rule_based_tags_rep, sentence_str = batch_

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
            return token_tensor, tag_tensor, mask_tensor, token_masks_tensor, gold_spans, pos_tensor, rule_based_tags_rep, sentence_str
        else:
            return token_tensor, tag_tensor, mask_tensor, token_masks_tensor, gold_spans, None, rule_based_tags_rep, sentence_str


class CascadeCoNLLReader(CoNLLReader):
    def read_data(self, data):
        dataset_name = data if isinstance(data, str) else 'dataframe'
        logger.info('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, _ in get_ner_reader(data=data):
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, coded_border_, coded_pos_, gold_spans_, mask = self.parse_line_for_ner(fields=fields)

            tokens_tensor = torch.tensor(tokens_sub_rep, dtype=torch.long)
            tag_tensor = torch.tensor(coded_ner_, dtype=torch.long)
            border_tensor = torch.tensor(coded_border_, dtype=torch.long)

            if self.use_pos_tag:
                pos_tensor = torch.tensor(coded_pos_, dtype=torch.long)
            else:
                pos_tensor = None

            token_masks_rep = torch.tensor(token_masks_rep)
            mask_rep = torch.tensor(mask)

            self.instances.append((tokens_tensor, mask_rep, token_masks_rep, gold_spans_, tag_tensor, border_tensor, pos_tensor, sentence_str))
            instance_idx += 1
        logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def parse_line_for_ner(self, fields):
        if self.mode == "train":
            # 动态MASK
            if self.use_dynamic_mask:
                for i, ner_tag in enumerate(fields[3]):
                    if ner_tag=="O" and random.random() < 0.15:
                        fields[0][i] = self.tokenizer.mask_token
        elif self.mode == "valid":
            pass
        else:
            assert self.mode == "test"
            # 预处理获取特征
            # 预处理获取特征
            if self.use_lemma is True or self.use_pos_tag is True:
                fields = self.preprocess_for_spacy(fields)

        if self.use_lemma is True:
            tokens_, ner_tags = fields[1], fields[3]
        else:
            tokens_, ner_tags = fields[0], fields[3]
        
        if self.use_pos_tag is True:
            pos_tags = fields[2]
        else:
            pos_tags = None

        sentence_str, tokens_sub_rep, ner_tags_rep, pos_tags_rep, token_masks_rep, mask = self.parse_tokens_for_ner(tokens_, ner_tags, pos_tags)
        # subwords, subword_idxs, subword_labels, subword_mask, subword_pos, mask
        gold_spans_ = extract_spans(ner_tags_rep, iob_tagging=self.iob_tagging)

        # 定义 实体 编码
        coded_ner_ = [self.label_to_id[tag] if tag in self.label_to_id else self.label_to_id['O'] for tag in ner_tags_rep]

        # 定义 边界 编码
        self.border_to_id = get_border_vocab_from_tag_vocab(self.label_to_id)
        coded_border_ = [ self.border_to_id[ tag[:1]+"-Border"] if tag!="O" else self.border_to_id['O'] for tag in ner_tags_rep]

        if self.use_pos_tag:
            coded_pos_ = [self.pos_tag_to_idx[pos] if pos in self.pos_tag_to_idx else self.pos_tag_to_idx['UNK'] for pos in pos_tags_rep]
        else:
            coded_pos_ = None
        return sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, coded_border_, coded_pos_, gold_spans_, mask

    def collate_batch_fn(self, batch):
        if len(batch[0]) != 8:
            print(batch[0])
            raise ValueError("what the fuck1")
        else:
            pass
        batch_ = list(zip(*batch))
        if len(batch_) != 8:
            print(batch_)
            raise ValueError("what the fuck2")
        else:
            pass
        tokens, masks, token_masks, gold_spans, tags, borders, poss, sentence_str = batch_

        max_len = max([len(token) for token in tokens])
        token_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.pad_token_id)
        tag_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.tag_to_id['O'])
        border_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.border_to_id['O'])
        
        if self.use_pos_tag:
            pos_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.pos_tag_to_idx['PAD'])
        
        mask_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
        token_masks_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)

        for i in range(len(tokens)):
            tokens_ = tokens[i]
            seq_len = len(tokens_)

            token_tensor[i, :seq_len] = tokens_
            tag_tensor[i, :seq_len] = tags[i]
            border_tensor[i, :seq_len] = borders[i]

            if self.use_pos_tag:
                pos_tensor[i, :seq_len] = poss[i]

            mask_tensor[i, :seq_len] = masks[i]
            token_masks_tensor[i, :seq_len] = token_masks[i]

        if self.use_pos_tag:
            return token_tensor, tag_tensor, border_tensor, mask_tensor, token_masks_tensor, gold_spans, pos_tensor, sentence_str
        else:
            return token_tensor, tag_tensor, border_tensor, mask_tensor, token_masks_tensor, gold_spans, None, sentence_str



class MrcCoNLLReader(CoNLLReader):
    def __init__(self, 
                 span_type = None,
                 **argv
                 ):
        super(MrcCoNLLReader, self).__init__(**argv)
        self.span_type = span_type

    def read_data(self, data):
        dataset_name = data if isinstance(data, str) else 'dataframe'
        logger.info('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, _ in get_ner_reader(data=data):
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            sentence_str, tokens_sub_rep, token_masks_rep, coded_start_, coded_end_, span_type_, coded_pos_, gold_spans_, mask = self.parse_line_for_ner(fields=fields)

            tokens_tensor = torch.tensor(tokens_sub_rep, dtype=torch.long)
            start_tensor = torch.tensor(coded_start_, dtype=torch.long)
            end_tensor = torch.tensor(coded_end_, dtype=torch.long)

            if self.use_pos_tag:
                pos_tensor = torch.tensor(coded_pos_, dtype=torch.long)
            else:
                pos_tensor = None

            token_masks_rep = torch.tensor(token_masks_rep)
            mask_rep = torch.tensor(mask)

            self.instances.append((tokens_tensor, mask_rep, token_masks_rep, gold_spans_, start_tensor, end_tensor, span_type_, pos_tensor, sentence_str))
            instance_idx += 1
        logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def parse_line_for_ner(self, fields):
        if self.mode == "train":
            # 动态MASK
            if self.use_dynamic_mask:
                for i, ner_tag in enumerate(fields[3]):
                    if ner_tag=="O" and random.random() < 0.15:
                        fields[0][i] = self.tokenizer.mask_token
        elif self.mode == "valid":
            pass
        else:
            assert self.mode == "test"
            # 预处理获取特征
            if self.use_lemma is True or self.use_pos_tag is True:
                fields = self.preprocess_for_spacy(fields)

        if self.use_lemma is True:
            tokens_, ner_tags = fields[1], fields[3]
        else:
            tokens_, ner_tags = fields[0], fields[3]
        
        if self.use_pos_tag is True:
            pos_tags = fields[2]
        else:
            pos_tags = None

        sentence_str, tokens_sub_rep, ner_tags_rep, pos_tags_rep, token_masks_rep, mask = self.parse_tokens_for_ner(tokens_, ner_tags, pos_tags)
        # subwords, subword_idxs, subword_labels, subword_mask, subword_pos, mask
        gold_spans_ = extract_spans(ner_tags_rep, iob_tagging=self.iob_tagging)

        # 定义 开始 和 结束
        _starts = [0] * len(tokens_sub_rep)
        _ends = [0] * len(tokens_sub_rep)
        for (i, j), span_type in gold_spans_.items():
            if span_type == self.span_type:
                _starts[i] = 1
                _ends[j] = 1

        _type = self.label_to_id["B-" + self.span_type]

        if self.use_pos_tag:
            coded_pos_ = [self.pos_tag_to_idx[pos] if pos in self.pos_tag_to_idx else self.pos_tag_to_idx['UNK'] for pos in pos_tags_rep]
        else:
            coded_pos_ = None
        return sentence_str, tokens_sub_rep, token_masks_rep, _starts, _ends, _type, coded_pos_, gold_spans_, mask

    def collate_batch_fn(self, batch):
        if len(batch[0]) != 9:
            print(batch[0])
            raise ValueError("what the fuck1")
        else:
            pass
        batch_ = list(zip(*batch))
        if len(batch_) != 9:
            print(batch_)
            raise ValueError("what the fuck2")
        else:
            pass
        tokens, masks, token_masks, gold_spans, starts, ends, _types, poss, sentence_str = batch_

        max_len = max([len(token) for token in tokens])
        token_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.pad_token_id)
        start_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(0)
        end_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(0)

        type_tensor = torch.tensor(_types, dtype=torch.long)

        if self.use_pos_tag:
            pos_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.pos_tag_to_idx['PAD'])
        
        mask_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
        token_masks_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)

        for i in range(len(tokens)):
            tokens_ = tokens[i]
            seq_len = len(tokens_)

            token_tensor[i, :seq_len] = tokens_
            start_tensor[i, :seq_len] = starts[i]
            end_tensor[i, :seq_len] = ends[i]

            if self.use_pos_tag:
                pos_tensor[i, :seq_len] = poss[i]

            mask_tensor[i, :seq_len] = masks[i]
            token_masks_tensor[i, :seq_len] = token_masks[i]
        if self.use_pos_tag:
            return token_tensor, start_tensor, end_tensor, mask_tensor, type_tensor, token_masks_tensor, gold_spans, pos_tensor, sentence_str
        else:
            return token_tensor, start_tensor, end_tensor, mask_tensor, type_tensor, token_masks_tensor, gold_spans, None, sentence_str

