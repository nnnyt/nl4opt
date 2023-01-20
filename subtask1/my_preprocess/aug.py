"""
    利用「任务2」中的信息和一些「额外的信息」, 增强任务1的数据
"""

from collections import defaultdict
import copy
import os
import json


def check2mkdir(file_path):
    "note: file_path must be like path/to/dir/filename"
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def read_json(open_path):
    load_data = json.load(open(open_path, 'r', encoding='utf-8'))
    print("[read_json] num = {}, open_path = {}".format(len(load_data), open_path))
    return load_data

def write_json(save_data, output_path):
    check2mkdir(output_path)
    with open(output_path, 'w+', encoding="utf-8") as f:
        f.write(json.dumps(save_data, indent=4, ensure_ascii=False))
    print("[write_json] num = {}, save_path = {}".format(len(save_data), output_path))

def read_line_json(open_path, error_handler="raise"):
    print("[read_line_json] start : {}".format(open_path))
    load_data = []
    i = 0
    with open(open_path, 'r', encoding="utf-8") as f:
        try:
            for line in f:
                load_data.append(json.loads(line))
                i += 1
        except Exception as e:
            if error_handler == "ignore":
                warnings.warn("[Warning] at line {}:\n{}\n".format(i, e))
            else:
                print("[Exception] at line {}:\n{}\n".format(i, e))
                raise Exception("[read_line_json] 出现错误")
    print("[read_line_json] num = {}, open_path = {}".format(len(load_data), open_path))
    return load_data

DATA_DIR = "data/generation_data/train.jsonl"
items = read_line_json(DATA_DIR)

_items_list = []
for item in items:
    for _key, _item in item.items():
        _item_new = copy.deepcopy(_item)
        _item_new.update({
            "qid": _key
        })
        _items_list.append(_item_new)
items = _items_list

# ----------------------------------------------------------------- #
# 修正数据
# ----------------------------------------------------------------- #
def rectify_percent(_items):
    for item in _items:
        pre_token = item["tokens"][0]
        for token in item["tokens"]:
            if token["text"] == "%":
                if pre_token["ner"] != "O" and token["ner"] == "O":
                    token["ner"] = F"I-{pre_token['ner'][2:]}"
            pre_token = token

# ------------------------------------------------------------------------------------ #
# 1. 补充词法信息
# ------------------------------------------------------------------------------------ #
def aug_detail(_items):
    # 1.1 合并BIO编码
    for idx, item in enumerate(_items):
        for token in item["tokens"]:
            token["ner"] = "O"
        spans = item["spans"]
        for span in spans:
            token_ner = span["label"]
            for i in range(span["token_start"], span["token_end"]+1):
                item["tokens"][i].update({
                    "ner": f"I-{token_ner}"
                })

            item["tokens"][span["token_start"]].update({
                "ner": f"B-{token_ner}"
            })
        # print(f"Finish {idx}")

    print("check _items.tokens: ", _items[0]["tokens"][0])

    # 1.2. 扩充词法知识
    import spacy
    nlp = spacy.load("en_core_web_sm")

    for idx, item in enumerate(_items):
        detail_tokens = nlp(item["document"])
        assert len(item["tokens"]) == len(detail_tokens)

        for token, detail_token in zip(item["tokens"], detail_tokens):
            assert token["text"] == detail_token.text
            token.update({
                "pos": detail_token.pos_,
                "lemma": detail_token.lemma_
            })
        # print(f"Finish {idx}")

    print("check tokens: ", _items[0]["tokens"][0])

    return _items


items_augmented_detail = copy.deepcopy(items)
items_augmented_detail = aug_detail(items_augmented_detail)


# ------------------------------------------------------------------------------------ #
# 2. 替换部分实体词, 和数字
# ------------------------------------------------------------------------------------ #
from nltk.corpus import wordnet
import random
import re
import warnings
import copy
from collections import defaultdict
from operator import itemgetter
num_words1 = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
num_words2 = ["once", "twice"]

def get_all_obj_words(span, tokens):
    ent_words = []
    for t in tokens[span["token_start"]: span["token_end"]+1]:
        ent_words.append(t["lemma"].lower())
    return ent_words

def get_overlap_max_obj_word(ents):
    word_2_num = defaultdict(int)
    for ent in ents:
        for w in ent:
            word_2_num[w] += 1
    data_order = sorted(word_2_num.items(),key=itemgetter(1),reverse=True)
    # [("word": num), ()]
    return data_order[0][0] # 只取重合次数最多次的词

def get_new_obj(old_ent):
    synonym = []
    antonym = []
    for syn in wordnet.synsets(old_ent):
        for lm in syn.lemmas():
            # 同义词
            synonym.append(lm.name())
            # 反义词
            if lm.antonyms():
                antonym.append(lm.antonyms()[0].name())
    synonym = synonym[1:]
    # 随机选择一个同义（反义）词
    if synonym + antonym:
        ent_synonym = random.choice(synonym + antonym)
    else:
        ent_synonym = old_ent
    return ent_synonym

# 替换 BOJ 实体 - 词根
def replace_new_obj(span_tokens, old_ent, new_ent):
    for t in span_tokens:
        if t["lemma"].lower() == old_ent:
            t["text"] = new_ent.title() if new_ent[0].isupper() else new_ent
            t["lemma"] = new_ent.lower()

# 扩充 BOJ 实体 - 词组（加上 total 等词）\
obj_pattens = [
    "number of ",
    "total number of ",
    "amount of ",
    "total amount of ",
    "total ",
]

# ------------------------------------------------------------ #
# 3.1 替换部分实体词, 和数字
# ------------------------------------------------------------ #
import spacy
nlp = spacy.load("en_core_web_sm")


def aug_2(_items_augmented_2):
    num_replace_limit = 0
    num_replace_param = 0
    num_replace_var = 0
    num_replace_obj = 0
    num_enlarge_obj = 0
    num_enlarge_boj_dict = defaultdict(int)

    weird_items = []

    for item in _items_augmented_2:
        old_item = copy.deepcopy(item)
        # 变量组
        var_mention_to_first_var = old_item["var_mention_to_first_var"]
        var_mention_to_first_var.update({
            var: var for var in old_item["vars"]
        })
        first_var_to_mentions = {
             var: [var] for var in old_item["vars"]
        }
        first_var_to_mentions.update(old_item["first_var_to_mentions"])
        
        # TODO: 是否可以使用“变量 1”和“变量 2”这样的特殊符号指导？
        obj_ents = []
        for span in item["spans"]:
            if span["label"]=="OBJ_NAME":
                obj_ents.append(get_all_obj_words(span, item["tokens"])) # [[a ent], [b ent], [c ent]]

        old_obj_ent = get_overlap_max_obj_word(obj_ents) # ent
        new_obj_ent = get_new_obj(old_obj_ent) # ent
        
        do_replace_obj = True
        if len(old_obj_ent) == 0 or len(new_obj_ent) == 0:
            # 实体替换出错，跳过
            warnings.warn("实体替换出错: ", old_obj_ent, "  -  ", new_obj_ent)
            do_replace_obj = False

        weird = False
        for span in item["spans"]:
            ents = [t for t in item["tokens"][span["token_start"]: span["token_end"]+1]]
            if do_replace_obj and span["label"] == "OBJ_NAME":
                """ 目标实体 """
                replace_new_obj(ents, old_obj_ent, new_obj_ent)
                # print(f"====replace ent: {old_obj_ent} -> {new_obj_ent}")
                num_replace_obj += 1
            elif span["label"] in ["PARAM", "LIMIT"]:
                """ 替换数字 """
                if random.random() < 0.5:
                    # ========================= 
                    if span["label"] == "PARAM":
                        num_replace_param += 1
                    else:
                        num_replace_limit += 1
                    
                    if "text" not in span:
                        warnings.warn('"text" not in span')
                        continue
                    if re.match(r"^\d+\.\d+%$", span["text"]):
                        # 小数%
                        assert span["token_start"] + 1 == span["token_end"]
                        token = item["tokens"][span["token_start"]]
                        token["text"] = str(random.randint(0, 99)) + f"{random.random():.2}"
                    elif re.match(r"^\d+%$", span["text"]):
                        # 整数%
                        assert span["token_start"] + 1 == span["token_end"]
                        token = item["tokens"][span["token_start"]]
                        token["text"] = str(random.randint(0, 99))
                    elif re.match(r"^\d+\.\d+$", span["text"]):
                        # 小数
                        assert span["token_start"] == span["token_end"]
                        token = item["tokens"][span["token_start"]]
                        token["text"] = str(random.randint(int(float(token["text"])*0.8),
                                                       int(float(token["text"])*1.2))
                                            ) + f"{random.random():.2}"      
                    elif re.match(r"^\d+$", span["text"]):
                        # 整数
                        assert span["token_start"] == span["token_end"]
                        token = item["tokens"][span["token_start"]]
                        num = int(token["text"])
                        token["text"] = str(random.randint(int(num*0.8), int(num*1.2)))
                        """
                        if num < 10 and num > 0:
                            if random.random() < 0.7:
                                if num < 2:
                                    # 依概率  转译为英文
                                    token["text"] = num_words2[num-1]
                                elif num < 10:
                                    # 依概率  转译为英文
                                    token["text"] = num_words1[num-1]
                            else:
                                # 依概率 替换数字
                                token["text"] = random.randint(
                                    int(num*0.8), int(num*1.2))
                        else:
                            # 依概率 替换数字
                            token["text"] = random.randint(
                                int(num*0.8), int(num*1.2))
                        """
                    else:
                        weird = True
                        # 粗浅处理 1,600 / 4 times / ...  等情况
                        old_span = span["text"]
                        new_span = ""
                        for token in item["tokens"][span["token_start"]: span["token_end"]+1]:
                            new_token = []
                            for _char in token["text"]:
                                if _char.isdigit() and _char != "0":
                                    new_token.append(str(random.randint(1, 9)))
                                else:
                                    new_token.append(_char)
                            new_token = "".join(new_token)
                            
                            for num_word in num_words1:
                                # 
                                if num_word in new_token:
                                    new_token = new_token.replace(num_word, num_words1[random.randint(0, 8)])
                                    break
                            
                            new_span += " " + new_token
                            token["text"] = new_token
                        warnings.warn(f"num span besides exception: {old_span} -> {new_span}")
                    
                    token["text"] = str(token["text"])
                    # assert token["text"]
        # 交换 变量
        tokens = copy.deepcopy(item["tokens"])
        new_tokens = []
        prev_span = ""
        pre_ner = "O"
        for i, token in enumerate(tokens+[{"text": ".", "lemma": ".", "pos": "PUNC", "ner": "O"}]):
            if token["ner"][2:] != "VAR" or (token["ner"]=="B-VAR" and pre_ner[2:] == "VAR"):
                if prev_span:
                    span_var = prev_span.strip() # 去除最后一个空格
                    first_var = var_mention_to_first_var[ span_var ]
                    new_var_idx = 1 - old_item["vars"].index(first_var)
                    if new_var_idx < 0:
                        warnings.warn("VAR number is bigger than 2 !")
                        prev_span = ""
                        
                        new_tokens.append(token)
                        if token["ner"]=="B-VAR" and pre_ner[2:] == "VAR":
                            prev_span += token["text"] + (" " if token["ws"] else "")
                        pre_ner = token["ner"]
                        continue
                    new_var = old_item["vars"][ new_var_idx ]
                    new_var_mentions = first_var_to_mentions[ new_var ]

                    final_new_idx = random.randint(0, len(new_var_mentions)-1)
                    final_new_mention = new_var_mentions[ final_new_idx ]  
                    # 加入新替换的变量
                    new_span = [
                        {
                            "text": tok.text, "start": -1, "end": -1, "id": -1, "ws": True, "disabled": False,
                            "pos": tok.pos_, "lemma": tok.lemma_, "ner": "I-VAR"
                        } for tok in nlp(final_new_mention)
                    ]
                    new_span[0]["ner"] = "B-VAR"
                    new_tokens.extend(new_span)
                    # print(f"______replace {span_var} -> {final_new_mention}")
                    prev_span = ""
                    num_replace_var += 1
                new_tokens.append(token)
                if token["ner"]=="B-VAR" and pre_ner[2:] == "VAR":
                    prev_span += token["text"] + (" " if token["ws"] else "")
                pre_ner = token["ner"]
            else:
                prev_span += token["text"] + (" " if token["ws"] else "")
                pre_ner = token["ner"]
        item["tokens"] = new_tokens        
        # 更新 题目
        item["document"] = " ".join([t["text"] for t in item["tokens"]])
        if weird is True:
            weird_items.append(old_item)
    
    total_num = len(_items_augmented_2)
    print(f"replace var : {num_replace_var} / {total_num} = {num_replace_var/total_num}")
    print(f"replace obj : {num_replace_obj} / {total_num} = {num_replace_obj/total_num}")
    print(f"enlarge obj : {num_enlarge_obj} / {total_num} = {num_enlarge_obj/total_num}")
    print("num_enlarge_boj_dict", num_enlarge_boj_dict)
    print()
    print(f"replace param : {num_replace_param} / {total_num} = {num_replace_param/total_num}")
    print(f"replace limit : {num_replace_limit} / {total_num} = {num_replace_limit/total_num}")
    print()
    print("weird_items num = ", len(weird_items))
    return _items_augmented_2, weird_items


items_augmented_2 = copy.deepcopy(items_augmented_detail)
items_augmented_2, weird_items = aug_2(items_augmented_2)


aug_items = []
len2 = len(items_augmented_2)
for item, item2 in zip(items_augmented_detail[:len2], items_augmented_2):
    # 让相似的数据增广在同一个batch
    aug_items.append(item)
    aug_items.append(item2)
aug_items.extend(items_augmented_detail[len2:1])


print("This final augmented items: num = ", len(aug_items))

# 获取最终数据
task_1_file = "data/train/train_aug_v3.2_new.txt"

with open(task_1_file, "w") as f:
    for item in aug_items:
        f.write("\n")
        for token in item["tokens"]:
            ner = token["ner"]
            text = token["text"]
            # f.write(f"{text}\t_\t_\t{ner}\n")
            pos = token["pos"]
            lemma = token["lemma"]
            f.write(f"{text}\t{lemma}\t{pos}\t{ner}\n")
    print("Finish writing ... for augmented items")
