import gzip
import itertools

def get_border_vocab_from_tag_vocab(label_to_id):
    border_to_id = dict()
    for key, _ in label_to_id.items():
        border_key = key[:2] + "Border" if key != "O" else "O"
        if border_key not in border_to_id:
            border_to_id[border_key] = len(border_to_id)
    return border_to_id

def get_ner_reader(data):
    fin = gzip.open(data, 'rt') if data.endswith('.gz') else open(data, 'rt', encoding="utf-8")
    for is_divider, lines in itertools.groupby(fin, _is_divider):
        if is_divider:
            continue
        # lines = [line.strip() for line in lines]
        lines = [line.strip().replace('\u200d', '').replace('\u200c', '').replace('\u200b', '') for line in lines]

        metadata = lines[0].strip() if lines[0].strip().startswith('# id') else None
        fields = [line.split() for line in lines if not line.startswith('# id')]
        
        fields = [field for field in fields if len(field) == 4]
        
        fields = [list(field) for field in zip(*fields)]
        
        yield fields, metadata

# 处理子词的tag和mask
def _assign_ner_tags(ner_tag, rep_, iob_tagging="conll"):
    '''
    Changing the token_masks so that only the first sub_word of a token has a True value, while the rest is False. This will be used for storing the predictions.
    :param ner_tag:
    :param rep_:
    :return:
    '''
    ner_tags_rep = []

    sub_token_len = len(rep_)
    mask_ = [False] * sub_token_len

    if len(mask_):
        mask_[0] = True

    if iob_tagging == "conll":
        if ner_tag[0] == 'B':
            in_tag = 'I' + ner_tag[1:]
            ner_tags_rep.append(ner_tag)
            ner_tags_rep.extend([in_tag] * (sub_token_len - 1))
        else:
            ner_tags_rep.extend([ner_tag] * sub_token_len)
    elif iob_tagging == "conll_ob":
        modify_tag = ner_tag.replace("I-", "B-")
        ner_tags_rep.extend([modify_tag] * sub_token_len)
    elif iob_tagging == "bio_const_dir":
        # 处理首位
        modify_start_tag = ner_tag
        for _type in ['LIMIT', 'CONST_DIR', 'VAR', 'PARAM', 'OBJ_NAME', 'OBJ_DIR']:
            if _type == "CONST_DIR":
                continue
            # 将其他实体转化为 O
            if _type in modify_start_tag:
                modify_start_tag = "O"
        # 处理子词
        if modify_start_tag[0] == 'B':
            in_tag = 'I' + modify_start_tag[1:]
            ner_tags_rep.append(modify_start_tag)
            ner_tags_rep.extend([in_tag] * (sub_token_len - 1))
        else:
            ner_tags_rep.extend([modify_start_tag] * sub_token_len)
    
    elif iob_tagging == "bio_obj_name":
        # 处理首位
        modify_start_tag = ner_tag
        for _type in ['LIMIT', 'CONST_DIR', 'VAR', 'PARAM', 'OBJ_NAME', 'OBJ_DIR']:
            if _type == "OBJ_NAME":
                continue
            # 将其他实体转化为 O
            if _type in modify_start_tag:
                modify_start_tag = "O"
        # 处理子词
        if modify_start_tag[0] == 'B':
            in_tag = 'I' + modify_start_tag[1:]
            ner_tags_rep.append(modify_start_tag)
            ner_tags_rep.extend([in_tag] * (sub_token_len - 1))
        else:
            ner_tags_rep.extend([modify_start_tag] * sub_token_len)
    elif iob_tagging == "bio_var":
        # 处理首位
        modify_start_tag = ner_tag
        for _type in ['LIMIT', 'CONST_DIR', 'VAR', 'PARAM', 'OBJ_NAME', 'OBJ_DIR']:
            if _type == "VAR":
                continue
            # 将其他实体转化为 O
            if _type in modify_start_tag:
                modify_start_tag = "O"
        # 处理子词
        if modify_start_tag[0] == 'B':
            in_tag = 'I' + modify_start_tag[1:]
            ner_tags_rep.append(modify_start_tag)
            ner_tags_rep.extend([in_tag] * (sub_token_len - 1))
        else:
            ner_tags_rep.extend([modify_start_tag] * sub_token_len)
    else:
        raise Exception("conll_ob or conll ERROR")

    return ner_tags_rep, mask_


def _assign_ner_tags_for_rule_tags(rule_tag, rep_):
    '''
    因为 rule_tag 可能为 None, 因此不能使用上面的 _assign_ner_tags 来为规则的子词赋标签
    '''
    rule_tags_rep = []

    sub_token_len = len(rep_)
    mask_ = [False] * sub_token_len

    if len(mask_):
        mask_[0] = True

    if not rule_tag:
        rule_tags_rep.extend([None] * sub_token_len)
    elif rule_tag[0] == 'B':
        in_tag = 'I' + rule_tag[1:]
        rule_tags_rep.append(rule_tag)
        rule_tags_rep.extend([in_tag] * (sub_token_len - 1))
    elif rule_tag[0] == 'I':
        rule_tags_rep.extend([rule_tag] * sub_token_len)
    else:
        rule_tags_rep.extend([rule_tag] * sub_token_len)
    return rule_tags_rep, mask_

def extract_spans(tags, iob_tagging="conll"):
    cur_tag = None
    cur_start = None
    gold_spans = {}

    def _save_span(_cur_tag, _cur_start, _cur_id, _gold_spans):
        if _cur_start is None:
            return _gold_spans
        _gold_spans[(_cur_start, _cur_id - 1)] = _cur_tag  # inclusive start & end, accord with conll-coref settings
        return _gold_spans

    # iterate over the tags
    for _id, nt in enumerate(tags):
        indicator = nt[0]
        if indicator == 'B':
            # if iob_tagging=="conll_ob" and nt[2:] == cur_tag:
            #     continue
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt[2:]
            pass
        elif indicator == 'I':
            # do nothing
            pass
        elif indicator == 'O':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_tag = 'O'
            cur_start = _id
            pass
    _save_span(cur_tag, cur_start, _id + 1, gold_spans)
    return gold_spans


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True

    first_token = line.split()[0]
    if first_token == "-DOCSTART-":  # or line.startswith('# id'):  # pylint: disable=simplifiable-if-statement
        return True

    return False


def get_tags(tokens, tags, tokenizer=None, start_token_pattern='▁'):
    tag_results = [], []
    index = 0
    tokens = tokenizer.convert_ids_to_tokens(tokens)
    for token, tag in zip(tokens, tags):
        if token == tokenizer.pad_token:
            continue

        if index == 0:
            tag_results.append(tag)

        elif token.startswith(start_token_pattern) and token != '▁́':
            tag_results.append(tag)
        index += 1

    return tag_results