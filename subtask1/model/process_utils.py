import copy

def handle_times_behind_num(tokenizer, batch_pred_spans, batch_tokens, batch_pred_tags, mode="test", verbose=False):

    batch_new_spans =[]

    token_times_id = tokenizer.encode("times")[-2]

    for pred_spans, tokens in zip(batch_pred_spans, batch_tokens):
        new_spans = dict()
        
        for span, _type in pred_spans.items():
            s, e = span
            if _type in ["LIMIT", "PARAM"] and tokens[e] == token_times_id:
                # 除去LIMIT前的times
                new_e = e-1
                new_e = new_e if s <= new_e else e
                new_spans[(s, new_e)] = _type
                print("1.2-1: remove times for limit")
            # elif _type == "PARAM" and tokens[e+1] == token_times_id:
            #     # 加上params后的times
            #     new_e = e + 1
            #     new_spans[(s, new_e)] = _type
            #     print("1.2-2: add times for param")
            else:
                new_spans[span] = _type
        batch_new_spans.append(new_spans)
    return batch_new_spans

def append_postfix_behind_num(tokenizer, batch_pred_spans, batch_tokens, batch_pred_tags, mode="test", verbose=False):
    token_times_id = tokenizer.encode("times")[-2]
    token_percent1_id = tokenizer.encode("%")[-2]
    token_percent2_id = tokenizer.encode("percent")[-2]
    token_puncuation_ids = [tokenizer.encode(",")[-2], tokenizer.encode(".")[-2]]
    
    # postfixs = [token_times_id, token_percent_id]
    postfixs =  [token_percent1_id, token_percent2_id]
    batch_new_spans = []
    for pred_spans, tokens, tags in zip(batch_pred_spans, batch_tokens, batch_pred_tags):
        new_spans = dict()
        
        for span, _type in pred_spans.items():
            s, e = span
            if _type in ["LIMIT", "PARAM"] and tokens[e+1] in postfixs:
                punc_prev = s-1
                punc_next = e+1
                while punc_prev > 0:
                    if tokens[punc_prev] in token_puncuation_ids:
                        break
                    punc_prev -= 1
                while punc_next < len(tokens):
                    if tokens[punc_next] in token_puncuation_ids:
                        break
                    punc_next += 1
                sent_other_tags = tags[punc_prev: s] + tags[e+1: punc_next]
                if _type == "LIMIT" and "B-CONST_DIR" not in sent_other_tags:
                    """ 约束 LIMIT"""
                    rec_type = "PARAM"
                elif _type == "PARAM" and "B-CONST_DIR" in tags[s-4: e+4]:
                    """ 约束 PARAM"""
                    rec_type = "LIMIT"
                else:
                    rec_type = _type
                
                print(f"1.2: add {tokens[e+1]} behind num -> {tokens[s: e+2]} | ", tokenizer.decode(tokens[s: e+2]))
                new_spans[(s, e+1)] = rec_type
            else:
                new_spans[(s, e)] = _type

        batch_new_spans.append(new_spans)
    return batch_new_spans

# CONST_DIR 补充 或 扩展
def append_must_be_or_needs_before_const_dir_or_param(tokenizer, batch_pred_spans, batch_tokens, batch_pred_tags, mode="test", verbose=False):
    token_must_id = tokenizer.encode("must")[-2]
    token_must_be_id = tokenizer.encode("must be")[1: -1]
    token_needs_id = tokenizer.encode("needs")[-2]

    token_available_id = tokenizer.encode("available")[-2] # TOD TEST
    token_more_id = tokenizer.encode("more")[-2]
    token_has_id = tokenizer.encode("has")[-2]
    token_puncuation_ids = [tokenizer.encode(",")[-2], tokenizer.encode(".")[-2]]

    batch_new_spans = []
    for pred_spans, tokens, tags in zip(batch_pred_spans, batch_tokens, batch_pred_tags):
        new_spans = dict()
        
        for span, _type in pred_spans.items():
            s, e = span
            token_prefix = tokens[:s]

            # 处理 needs
            """ TO TEST """
            # if _type == "LIMIT" and token_needs_id in tokens[s-3: s]:
            #     # limit/all  = 10/11
            #     # num 前方的 needs xxx 必为 CONST_DIR
            #     needs_start = tokens[s-3: s].index(token_needs_id) + s-3
            #     new_spans[(needs_start, s-1)] = "CONST_DIR"
            #     new_spans[(s, e)] = _type
            #     print(f"[TEST] post2-1: add needs ... before PARAM and LIMIT -> {tokens[needs_start: s]} | ", tokenizer.decode(tokens[needs_start: s]))

            # 处理 must
            if _type == "CONST_DIR" and token_prefix[-1] == token_must_id and tokens[s:s+2] != tokenizer.encode("not be")[1:-1]:
                # 扩展 must CONST_DIR
                new_spans[(s-1, e)] = _type
                print(f"[TEST] post2-2: add must before CONST_DIR -> {tokens[s-1: e+1]} | ", tokenizer.decode(tokens[s-1: e+1]))
            elif _type == "PARAM" and token_prefix[-2:] == token_must_be_id:
                # param 前的 must be  必为 CONST_DIR
                new_spans[(s-2, s-1)] = "CONST_DIR"
                new_spans[(s, e)] = _type
                print(f"[TEST] post2-3: add must be before PARAM -> {tokens[s-2: e+1]} | ", tokenizer.decode(tokens[s-2: e+1]))
            
            # # 处理 available
            elif token_available_id in tokens[s: e]:
                """ 去除误判的 availble """
                punc_prev = s-1
                punc_next = e+1
                while punc_prev > 0:
                    if tokens[punc_prev] in token_puncuation_ids:
                        break
                    punc_prev -= 1
                while punc_next < len(tokens):
                    if tokens[punc_next] in token_puncuation_ids:
                        break
                    punc_next += 1
                sent_other_tags = tags[punc_prev: s] + tags[e+1: punc_next]
                if "B-CONST_DIR" in sent_other_tags:
                    new_spans[s, e] = "O"
                else:
                    new_spans[(s, e)] = _type
                
                # elif _type in ["PARAM", "LIMIT"] and token_available_id in tokens[s-5: s+4]:
                # # 条件：没有其他 CONST_DIR
                # token_neighbor, tag_neighbor = tokens[s-5: e+4], tags[s-5: e+4]
                # if "B-CONST_DIR" not in tag_neighbor and "I-CONST_DIR" not in tag_neighbor:
                #     # 补全可能的 available
                #     idx_start = token_neighbor.index(token_available_id) + s-5
                #     new_spans[(idx_start, idx_start)] = "CONST_DIR"

                #     new_spans[(s, e)] = _type
                #     print(f"[TEST] post2-4: add availale ... before PARAM and LIMIT -> {tokens[idx_start: s+4]} | ", tokenizer.decode(tokens[idx_start: s+4]))
                # else:
                #     new_spans[(s, e)] = _type
            
            # # 处理 more
            # elif _type == "VAR" and token_prefix[-1] == token_more_id:
            #     # 仅当 当前句子中没有 CONST_DIR 时， 将 more 补充为 const_dir
            #     punc_prev = s-1
            #     punc_next = e+1
            #     while punc_prev > 0:
            #         if tokens[punc_prev] in token_puncuation_ids:
            #             break
            #         punc_prev -= 1
            #     while punc_next < len(tokens):
            #         if tokens[punc_next] in token_puncuation_ids:
            #             break
            #         punc_next += 1
                # sent_tokens, sent_tags = tokens[punc_prev: punc_next], tags[punc_prev: punc_next]

            #     if ("B-CONST_DIR" not in sent_tags and  "I-CONST_DIR" not in sent_tags):
            #         if("B-LIMIT" in sent_tags or "I-LIMIT" in sent_tags or "B-PARAM" in sent_tags or "I-PARAM" in sent_tags):
            #             # var 前的 more 在没有其他 const_dir 并且此句话中有 PARAM 或 LIMIT 时，则此情况为漏掉的 const_dir
            #             new_spans[(s-1, s-1)] = "CONST_DIR"
            #         else:
            #             # 去掉 误判的 more var1 than var2
            #             new_spans[(s-1, s-1)] = "O"
            #         new_spans[(s, e)] = _type
            #         print(f"[TEST] post2-5: add more before VAR -> {tokens[s-1: e+1]} | ", tokenizer.decode(tokens[s-1: e+1]))
            #     else:
            #         new_spans[(s, e)] = _type
            
            # 处理 has
            elif _type == "LMIT" and token_has_id in tokens[s-6: e+6]:
                # limit 附近的 has 在没有其他 const_dir 时，则此情况为漏掉的 const_dir
                token_neighbor, tag_neighbor = tokens[s-6: e+6], tags[s-6: e+6]
                if ("B-CONST_DIR" not in tag_neighbor and  "I-CONST_DIR" not in tag_neighbor):
                    idx_start = token_neighbor.index(token_has_id) + s-6
                    new_spans[(idx_start, idx_start)] = "CONST_DIR"
                    new_spans[(s, e)] = _type
                    print(f"[TEST] post2-6: add has before LMIT -> {tokens[idx_start: e+1]} | ", tokenizer.decode(tokens[idx_start: e+1]))
                else:
                    new_spans[(s, e)] = _type
            else:
                new_spans[(s, e)] = _type
            
        batch_new_spans.append(new_spans)
    return batch_new_spans

def remove_for_or_other_behind_const_dir(tokenizer, batch_pred_spans, batch_tokens, batch_pred_tags, mode="test", verbose=False):
    token_for_id = tokenizer.encode("for")[1]
    token_available_id = tokenizer.encode("available")[1]

    batch_new_spans = []
    for pred_spans, tokens in zip(batch_pred_spans, batch_tokens):
        new_spans = dict()
        
        for span, _type in pred_spans.items():
            s, e = span

            # if _type == "CONST_DIR" and s<e and tokens[e+1]== token_for_id:
            #     new_spans[(s, e-1)] = _type
            #     print(f"[TEST] post2.3: remove for behind CONST_DIR -> {tokens[s: e]} | ", tokenizer.decode(tokens[s: e]))
            
            if _type == "CONST_DIR" and s<e and tokens[s] == token_available_id:
                # availble 后不接 for 等词, 一般单独出现
                new_spans[(s, s)] = _type
                print(f"[TEST] post2.2: remove (for) behind availble as CONST_DIR -> {tokens[s: e]} | ", tokenizer.decode(tokens[s: e]))
            else:
                new_spans[(s, e)] = _type
            
        batch_new_spans.append(new_spans)
    return batch_new_spans

def remove_fake_const_dir_before_var(tokenizer, batch_pred_spans, batch_tokens, batch_pred_tags, mode="test", verbose=False):
    token_more_id = tokenizer.encode("more")[1]

    batch_new_spans = []
    for pred_spans, tokens in zip(batch_pred_spans, batch_tokens):
        new_spans = dict()
        
        for span, _type in pred_spans.items():
            s, e = span
            token_prefix = tokens[:s]

            if _type == "VAR" and token_prefix[-1] == token_more_id and ("B-CONST_DIR" in token_prefix[-3:] or "I-CONST_DIR" in token_prefix[-3:]):
                # var 前的 more 在没有其他const_dir时，则此情况为多余的 const_dir
                new_spans[(s-1, s-1)] = "O"
                new_spans[(s, e)] = _type
                print(f"[TEST] post2.3: remove more before VAR -> {tokens[s-2: e+1]} | ", tokenizer.decode(tokens[s-2: e+1]))
            else:
                new_spans[(s, e)] = _type
        batch_new_spans.append(new_spans)
    return batch_new_spans




def append_prefix_before_obj_name(tokenizer, batch_pred_spans, batch_tokens, batch_pred_tags, mode="test", verbose=False):
    batch_new_spans =[]

    token_total_id = 3622
    token_the_id = 70

    amount_of_ids = [41170, 111]
    total_amount_of_ids = [3622, 41170, 111]

    number_of_ids = [14012, 111]
    total_number_of_ids = [3622, 14012, 111]

    unit_s_of_ids = [25072, 7, 111]
    total_unit_s_of_ids = [3622, 25072, 7, 111]

    for pred_spans, tokens in zip(batch_pred_spans, batch_tokens):
        new_spans = dict()
        
        for span, _type in pred_spans.items():
            s, e = span
            distance_end = len(tokens) - e
            if _type == "OBJ_NAME" and distance_end < 10 and s -4 >= 0:
                token_prefix = tokens[:s]

                """ WAY1: 扩展为 total number of XXX 模式 """
                if (token_prefix[-4:] == total_unit_s_of_ids):
                    # 扩展首部: total unit s of XXX 等
                    new_spans[(s-4, e)] = _type
                
                elif (token_prefix[-3:] == total_amount_of_ids 
                        or token_prefix[-3:] == total_number_of_ids):
                    # 扩展首部: total number of XXX 等
                    new_spans[(s-3, e)] = _type
                
                elif (token_prefix[-3:] == unit_s_of_ids):
                    # 扩展首部: unit s of XXX 等
                    new_spans[(s-3, e)] = _type

                elif token_prefix[-2:] == amount_of_ids or token_prefix[-2:] == number_of_ids:
                    # 扩展首部: number of XXX 等
                    new_spans[(s-2, e)] = _type

                elif token_prefix[-1] == token_total_id:
                    # 扩展首部: total (number of) XXX
                    new_spans[(s-1, e)] = _type
                
                elif tokens[s] == token_the_id:
                    # 除去首部的 the
                    new_spans[(s+1, e)] = _type
                else:
                    new_spans[span] = _type


                """ WAY2: 扩展为 number of XXX 模式 """
                # if (token_prefix[-3:] == unit_s_of_ids):
                #     # 扩展首部: unit s of XXX 等
                #     new_spans[(s-3, e)] = _type
                # elif token_prefix[-2:] == amount_of_ids or token_prefix[-2:] == number_of_ids:
                #     # 扩展首部: number of XXX 等
                #     new_spans[(s-2, e)] = _type
                
                # elif tokens[s: s+2] == [token_the_id, token_total_id]:
                #     # 去掉首部的 the total (XXX)
                #     new_spans[(s+2, e)] = _type
                # elif tokens[s] == token_total_id:
                #     # 去掉首部的 total (XXX)
                #     new_spans[(s+1, e)] = _type
                # elif tokens[s] == token_the_id:
                #     # 去掉首部的 the (XXX)
                #     new_spans[(s+1, e)] = _type
                # else:
                #     new_spans[span] = _type

                """ WAY3: 优先 number of XXX 和 the total number of XXX """
                # if (token_prefix[-4:] == total_unit_s_of_ids):
                #     # 扩展首部: (the total unit s of) XXX 等
                #     new_spans[(s-4, e)] = _type
                
                # elif (token_prefix[-3:] == total_amount_of_ids 
                #         or token_prefix[-3:] == total_number_of_ids):
                #     # 扩展首部: (the total number of) XXX 等
                #     new_spans[(s-3, e)] = _type

                # elif (token_prefix[-3:] == unit_s_of_ids):
                #     # 扩展首部: (unit s of) XXX 等
                #     new_spans[(s-3, e)] = _type
                # elif token_prefix[-2:] == amount_of_ids or token_prefix[-2:] == number_of_ids:
                #     # 扩展首部: (number of) XXX 等
                #     new_spans[(s-2, e)] = _type
                
                # elif tokens[s] == token_total_id:
                #     # 去掉首部的 total (XXX)
                #     new_spans[(s+1, e)] = _type
                # else:
                #     new_spans[span] = _type

            else:
                new_spans[span] = _type

        batch_new_spans.append(new_spans)
    return batch_new_spans


def rectify_obj_name_by_trigger(tokenizer, batch_pred_spans, batch_tokens, batch_pred_tags, mode="test", verbose=False):
    # if pred number of XXX as OBJ_NAME, then XXX before is OBJ_NAME
    batch_new_spans = []
    token_total_id = 3622
    token_the_id = 70

    amount_of_ids = [41170, 111]
    total_amount_of_ids = [3622, 41170, 111]

    number_of_ids = [14012, 111]
    total_number_of_ids = [3622, 14012, 111]

    unit_s_of_ids = [25072, 7, 111]
    total_unit_s_of_ids = [3622, 25072, 7, 111]

    for pred_spans, tokens, pred_tags in zip(batch_pred_spans, batch_tokens, batch_pred_tags):
        new_spans = copy.deepcopy(pred_spans)

        may_be_obj_name = []         # 获取 可能 的OBJ_NAME
        last_obj_name_pos = len(tokens) 
        for span, _type in pred_spans.items():
            s, e = span
            distance_end = len(tokens) - e
            if _type == "OBJ_NAME" and distance_end < 10:
                """ WAY2: 根据提示词 （如 number of XXX ) 找到可能的 OBJ_NAME (如 XXX ) """
                if tokens[s: s+3] == unit_s_of_ids:
                    may_be_obj_name = tokens[s+3: e+1]
                elif tokens[s: s+2] == amount_of_ids or tokens[s: s+2] == number_of_ids:
                    may_be_obj_name = tokens[s+2: e+1]
                if tokens[s: s+4] == total_unit_s_of_ids:
                    may_be_obj_name = tokens[s+4: e+1]
                elif tokens[s: s+3] == total_amount_of_ids or tokens[s: s+3] == total_number_of_ids:
                    may_be_obj_name = tokens[s+3: e+1]
                elif tokens[s] == token_total_id:
                    may_be_obj_name = tokens[s+3: e+1]
                else:
                    pass
                last_obj_name_pos = s
            else:
                pass


        _len = len(may_be_obj_name)
        if _len == 0:
            batch_new_spans.append(new_spans)
            continue
        
        if may_be_obj_name == tokenizer.encode("time")[1]:
            if tokenizer.encode("minutes")[1] in tokens:
                may_be_obj_name = [tokenizer.encode("minutes")[1]]
            elif tokenizer.encode("seconds")[1] in tokens:
                may_be_obj_name = [tokenizer.encode("seconds")[1]]
            elif tokenizer.encode("hours")[1] in tokens:
                may_be_obj_name = [tokenizer.encode("hours")[1]]
        else:
            batch_new_spans.append(new_spans)
            continue

        may_be_count = 0
        distance_end = len(tokens) - e
        # 仅仅处理 前面的 OBJ_NAME
        for i in range(last_obj_name_pos):
            # if tag == "O"
            if tokens[i: i+_len] == may_be_obj_name:
                may_be_count += 1
        
        if may_be_count>=2:
            for i in range(last_obj_name_pos):
                # 更正前面 未被标记的 OBJ_NAME 标签
                if pred_tags[i] == "O" and tokens[i: i+_len] == may_be_obj_name:
                    # 仅仅补充 PARAM 和 LIMIE 附近的 OBJ_NAME
                    add_ = False
                    for bias in range(-4, 5):
                        if (i + bias >= 0 and i + bias < len(tokens)
                                and  pred_tags[i + bias] in ["I-PARAM", "I-PARAM"]):
                            add_ = True
                    if add_:  # TODO : NO WORK
                        print(f"[TEST] post4: add OBJ_NAME -> {tokens[i: i + _len]} | ", tokenizer.decode(tokens[i: i + _len]))
                        new_spans[(i, i + _len - 1)] = "OBJ_NAME"    
        batch_new_spans.append(new_spans)
    return batch_new_spans


def handle_fake_or_missing_obj_name(tokenizer, batch_pred_spans, batch_tokens, batch_pred_tags, mode="test", verbose=False):
    batch_new_spans = []
    for pred_spans, tokens, tags in zip(batch_pred_spans, batch_tokens, batch_pred_tags):
        new_spans = copy.deepcopy(pred_spans)
        for span, _type in pred_spans.items():
            s, e = span
            token_neighbor, tag_neighbor = tokens[s-6:s] + tokens[e+1:e+6], tags[s-6:s] + tags[e+1:e+6]
            if _type == "OBJ_NAME" and "B-LIMIT" in tokens[s-2: e+3]:
                print("[TEST] post5-1: remove ", tokenizer.decode(tokens[s: e]))
                new_spans[(s,e)] = "O"
            # elif s < 10:
            #     new_spans[(s,e)] = "O"
            # elif _type == "OBJ_NAME" and (
            #     "B-VAR" not in tag_neighbor and "B-PARAM" not in tag_neighbor
            #     and "I-VAR" not in tag_neighbor and "I-PARAM" not in tag_neighbor
            #     ):
            #     new_spans[(s,e)] = "O"
            else:
                new_spans[(s,e)] = _type

        maybe_add = False
        for token, tag in zip(tokens, tags):
            if tag in ["I-OBJ_NAME", "B-OBJ_NAME"] and token == tokenizer.encode("time")[-2]:
                maybe_add = True
        
        if maybe_add:
            add_objs = [
                tokenizer.encode("seconds")[-2],
                tokenizer.encode("hours")[-2],
                tokenizer.encode("hour")[-2],
                tokenizer.encode("minutes")[-2],
            ]
            for idx, token in enumerate(tokens):
                if token in add_objs and tags[idx] == "O":
                    print("[TEST] post5-2: add ", tokenizer.decode([token]))
                    new_spans[(idx, idx)] = "OBJ_NAME"
        # else:
        #     print("[TEST] no seconds ... ")
        
        batch_new_spans.append(new_spans)
    return batch_new_spans


def post_process(tokenizer, batch_pred_spans, batch_tokens, batch_pred_tags, post_method=[1,2], mode="test", verbose=False):
    process = batch_pred_spans

    if 1 in post_method:
        # 1.1
        process = append_postfix_behind_num(tokenizer, process, batch_tokens, batch_pred_tags, mode="test", verbose=verbose) # %: o -> i
        # 1.2
        process = handle_times_behind_num(tokenizer, process, batch_tokens, batch_pred_tags, mode="test", verbose=verbose) # times: i -> o

    if 2 in post_method:
        # 2.1 扩展 must ( must be ) + CONST_DIR
        process = append_must_be_or_needs_before_const_dir_or_param(tokenizer, process, batch_tokens, batch_pred_tags, mode="test", verbose=verbose) # must CONST_DIR
        # 2.2 
        process = remove_for_or_other_behind_const_dir(tokenizer, process, batch_tokens, batch_pred_tags, mode="test", verbose=verbose) # CONST_DIR for
        # 2.3 约束 more + VAR 条件下，more 被错误识别为 CONST_DIR
        process = remove_fake_const_dir_before_var(tokenizer, process, batch_tokens, batch_pred_tags, mode="test", verbose=verbose) # more VAR

    if 3 in post_method:  
        # 完整标准化 对应的后处理操作
        if mode in ["test", "ensemble"]:
            process = append_prefix_before_obj_name(tokenizer, process, batch_tokens, batch_pred_tags, mode="test", verbose=verbose)
    
    if 4 in post_method:  
        # 4. 利用 trigger 扩充 OBJ_NAME
        process = rectify_obj_name_by_trigger(tokenizer, process, batch_tokens, batch_pred_tags, mode="test", verbose=verbose)
    
    if 5 in post_method:
        # 限制 obj_name
        process = handle_fake_or_missing_obj_name(tokenizer, process, batch_tokens, batch_pred_tags, mode="test", verbose=verbose)

    return process







if __name__ == "__main__":
    batch_pred_spans = [
        {(0, 1): "PARAM"},
        {(0, 2): "OBJ_NAME"},
        {(2, 2): "OBJ_NAME", (3,4): "PARAM"},
    ]
    batch_tokens = [
        [1, 20028],
        [41170, 111, 2],
        [41170, 111, 3, 4, 20028],
    ]

    batch_new_spans1 = handle_times_behind_num(batch_pred_spans, batch_tokens)
    print(batch_new_spans1)

    batch_new_spans2 = append_prefix_before_obj_name(batch_pred_spans, batch_tokens)
    print(batch_new_spans2)

    batch_new_spans2 = post_process(batch_pred_spans, batch_tokens)
    print(batch_new_spans2)
    