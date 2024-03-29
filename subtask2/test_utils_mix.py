import re
import difflib
import pickle
from selectors import EpollSelector
import tqdm
from torch.utils.data import DataLoader
from utils import *

import parsers
import scoring
from typing import Optional, Dict, List, Tuple
import os


def collate_score_declarations(pred_texts: List[str], 
                                gold_texts: List[str], 
                                doc_ids: List[str], 
                                order_mappings: List[Dict], 
                                print_errors=True) -> float:
    current_id = ''
    current_pred_problem = ''
    current_gold_problem = ''
    pred_problems = []
    gold_problems = []
    mappings = []

    pred_canonicals = []
    gold_canonicals = []

    # converts an output into canonical form and returns the canonical form along with the order mapping
    # please ensure that the order mapping is consistent between pred and gold or the columns may be incorrect in the canonical form
    def parse_convert(output: str, order_mapping: Dict) -> parsers.CanonicalFormulation:
        parser = parsers.ModelOutputXMLParser(print_errors=print_errors)
        parsed = parser.parse(output, order_mapping)
        return parsers.convert_to_canonical(parsed)
    
    # we append as we do our predictions on the declaration level, i.e., we keep appending declarations until we get to the next problem
    # loop assumes that same doc_ids are contiguous
    # models that predict the entire formulation at once should not use this loop
    for pred, gold, doc_id, order_mapping in zip(pred_texts, gold_texts,doc_ids,order_mappings):
        if current_id != doc_id:
            # append order mapping of new problem
            mappings.append(order_mapping)
            current_id = doc_id
            if current_pred_problem and current_gold_problem:
                # append texts of previous problem
                gold_problems.append(current_gold_problem)
                pred_problems.append(current_pred_problem)
                current_pred_problem = ''
                current_gold_problem = ''
        
        current_pred_problem += pred
        current_gold_problem += gold

    # append texts for last problem, don't need to do for order mapping as it will already have been appended
    gold_problems.append(current_gold_problem)
    pred_problems.append(current_pred_problem)

    for pred, gold, order_mapping in zip(pred_problems, gold_problems, mappings):
        # use gold's order mapping in prediction for consistency in producing canonical form
        gold_canonical = parse_convert(gold, order_mapping)
        pred_canonical = parse_convert(pred, order_mapping)
        pred_canonicals.append(pred_canonical)
        gold_canonicals.append(gold_canonical)

    return scoring.overall_score(
        [x.objective for x in pred_canonicals],
        [x.constraints for x in pred_canonicals],
        [x.objective for x in gold_canonicals],
        [x.constraints for x in gold_canonicals],
    )


def similarity(a: str, b: str):
    a = a.strip().lower()
    b = b.strip().lower()
    # ignore spaces with isjunk
    sm = difflib.SequenceMatcher(isjunk=lambda x: x in " \t", a=a, b=b)
    return sm.ratio()


def evaluate(tokenizer, tokenizer_c, 
                model, model_c, 
                dataset, dataset_c,
                epoch, 
                batch_num,
                use_gpu, 
                config, config_c,
                tqdm_descr="Dataset", 
                ckpt_basename = "",
                print_errors=True):

    progress = tqdm.tqdm(total=batch_num, ncols=75,
                         desc='{} {}'.format(tqdm_descr, ckpt_basename))
    gold_outputs, pred_outputs, input_tokens, doc_ids, documents, order_mappings = [], [], [], [], [], []
    pred_texts, gold_texts, gold_pred_pairs = [], [], []
    # metric = Rouge()
    measures = []
    for batch in DataLoader(dataset, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=dataset.collate_fn):
        progress.update(1)
        outputs = model.predict(batch, tokenizer, epoch=epoch)
        decoder_inputs_outputs = generate_decoder_inputs_outputs(batch, tokenizer, model, use_gpu,
                                                                config.max_position_embeddings)
        pred_outputs.extend(outputs['decoded_ids'].tolist())
        gold_outputs.extend(decoder_inputs_outputs['decoder_labels'].tolist())
        input_tokens.extend(batch.input_tokens)
        doc_ids.extend(batch.doc_ids)
        documents.extend(batch.document)
        order_mappings.extend(batch.order_mapping)
        pred_txt = [tokenizer.decode(x.tolist()) for x in outputs['decoded_ids']][0][4:]
        # print(pred_txt)
        gold_txt = [tokenizer.decode(x.tolist()) for x in decoder_inputs_outputs['decoder_labels']][0]
        pred_txt, gold_txt = pred_txt[3:-4], gold_txt[3:-4]
        pred_texts.append(pred_txt)
        gold_texts.append(gold_txt)
        # diff_metrics = metric.get_scores(pred_txt, gold_txt)
        diff_metrics = {}
        measures.extend(diff_metrics)

        gold_pred_pairs.append({
            "gold": gold_txt,
            "pred": pred_txt,
            "document": batch.document[0],
            "rouge": diff_metrics
        })
    progress.close()

    #######
    progress = tqdm.tqdm(total=batch_num, ncols=75,
                        desc='{} {}'.format(tqdm_descr, ckpt_basename))
    gold_outputs_c, pred_outputs_c, input_tokens_c, doc_ids_c, documents_c, order_mappings_c = [], [], [], [], [], []
    pred_texts_c, gold_texts_c, gold_pred_pairs_c = [], [], []

    for batch in DataLoader(dataset_c, batch_size=config_c.eval_batch_size,
                            shuffle=False, collate_fn=dataset_c.collate_fn):
        progress.update(1)
        outputs = model_c.predict(batch, tokenizer_c, epoch=epoch)
        decoder_inputs_outputs = generate_decoder_inputs_outputs(batch, tokenizer_c, model_c, use_gpu,
                                                                config_c.max_position_embeddings)
        pred_outputs_c.extend(outputs['decoded_ids'].tolist())
        gold_outputs_c.extend(decoder_inputs_outputs['decoder_labels'].tolist())
        input_tokens_c.extend(batch.input_tokens)
        doc_ids_c.extend(batch.doc_ids)
        documents_c.extend(batch.document)
        order_mappings_c.extend(batch.order_mapping)
        pred_txt_c = [tokenizer_c.decode(x.tolist()) for x in outputs['decoded_ids']][0][4:]
        # print(pred_txt)
        gold_txt_c = [tokenizer_c.decode(x.tolist()) for x in decoder_inputs_outputs['decoder_labels']][0]
        pred_txt_c, gold_txt_c = pred_txt_c[3:-4], gold_txt_c[3:-4]
        pred_texts_c.append(pred_txt_c)
        gold_texts_c.append(gold_txt_c)

        gold_pred_pairs_c.append({
            "gold": gold_txt_c,
            "pred": pred_txt_c,
            "document": batch.document[0],
        })
    progress.close()

    pred_texts_final, gold_texts_final, doc_ids_final, order_mappings_final, documents_final = [], [], [], [], []
    id_to_idx = {}
    for i, id in enumerate(doc_ids):
        if id not in id_to_idx:
            id_to_idx[id] = [i]
        else:
            id_to_idx[id].append(i)
    id_to_idx_c = {}
    for i, id in enumerate(doc_ids_c):
        if id not in id_to_idx_c:
            id_to_idx_c[id] = [i]
        else:
            id_to_idx_c[id].append(i)

    for id in id_to_idx.keys():
        idx_list = id_to_idx[id]
        idx_list_c = id_to_idx_c[id]

        for i in idx_list:
            if 'OBJ_DIR' in pred_texts[i]:    #pred_texts是span-in模型预测出来的
                pred_texts_final.append(pred_texts[i])
                gold_texts_final.append(gold_texts[i])
                doc_ids_final.append(id)
                order_mappings_final.append(order_mappings[i])
                documents_final.append(documents[i])
        for i in idx_list_c:
            if 'CONST_DIR' in pred_texts_c[i]:  #pred_texts是attach模型预测出来的
                pred_texts_final.append(pred_texts_c[i])
                gold_texts_final.append(gold_texts_c[i])
                doc_ids_final.append(id)
                order_mappings_final.append(order_mappings_c[i])
                documents_final.append(documents_c[i])
    # print(len(pred_texts_final), len(pred_texts), len(pred_texts_c))

    ####
    # add post-process here!!
    # rule1: 错误的不等式方向
    for i, r in enumerate(pred_texts_final):
        if 'larger' in r or 'must exceed' in r or 'need' in r or '<CONST_DIR> process </CONST_DIR>' in r:
            pred_texts_final[i] = r.replace('LESS_OR_EQUAL', 'GREATER_OR_EQUAL')

    # rule2: 修正错误的数字
    for i, r in enumerate(pred_texts_final):
        doc = documents_final[i]
        doc_num = re.findall(r"\d+[\.\,]?\d*", doc)
        for j, n in enumerate(doc_num):
            if n[-1] == '.' or n[-1] == ',':
                doc_num[j] = n[:-1]
        num = re.findall(r"\d+[\.\,]?\d*", r)
        for j, n in enumerate(num):
            real_value = eval(n.replace(',', ''))
            if real_value > 10 and n not in doc_num:
                best_similarity = 0
                best_match = None
                for dn in doc_num:
                    sim = similarity(dn, n)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match = dn
                if best_match:
                    pred_texts_final[i] = r.replace(n, best_match)
                    num[j] = best_match

    ######
    accuracy = collate_score_declarations(pred_texts_final, gold_texts_final, doc_ids_final, order_mappings_final, print_errors)
    # avg_metric = metric.get_scores(pred_texts, gold_texts, avg=True)
    avg_metric = {}

    result = {
        'accuracy': accuracy,
        'pred_outputs': pred_outputs,
        'gold_outputs': gold_outputs,
        'input_tokens': input_tokens,
        'doc_ids': doc_ids_final,
        'documents': documents_final,
        'pred_texts': pred_texts_final,
        'gold_texts': gold_texts_final,
        'gold_pred_pairs': gold_pred_pairs,
        'rouge': avg_metric
    }
    # print()
    # print(result['gold_pred_pairs'])
    # print()
    # print(result['doc_ids'])

    return result
