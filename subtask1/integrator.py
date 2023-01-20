import os
import torch

print("torch.cuda.is_available() = ", torch.cuda.is_available(), " |  device_count  ", torch.cuda.device_count())
# os.environ["CUDA_VISIBLE_DEVICES"]

from pytorch_lightning import seed_everything


from utils.utils import get_reader, parse_args_product, get_tagset, read_json, write_eval_performance, occumpy_mem
from transformers import PretrainedConfig
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup
import pandas as pd

from model.model import NerModel
from model.model_cascade import CascadeNerModel
from model.model_mrc import MRCNerModel
from model.trainer import CustomTrainer
from utils.metric import SpanF1
from tqdm import tqdm
from collections import Counter
from utils.reader_utils import extract_spans


def write_for_leaderboard(results, out_filename):
    ''' write the micro averaged F1 score to results.out '''
    open(out_filename, 'wt').write(str(results["micro@F1"]))


class IntegratorPredictor(object):
    def __init__(self, model_dirs, models, test_dataloader, method="span-based", gpu="cuda:0"):
        self.model_dirs = model_dirs
        self.models = models
        self.test_dataloader = test_dataloader
        self.method = method
        assert method in ["span-based", "tag-based", "logit-based"]
        self.mix_span_f1 = SpanF1()
        self.excel_metric_results = []
        self.gpu = gpu

    def test(self, use_tongxiao_post=True, use_post=True, post_method=[1,2], online_submission_test=False):
        print(f"DEBUG IntegratorPredictor: use_post = {use_post}, post_method={post_method}")
        prediction_results = []
        metric_results = []
        
        use_post_name = "post_"
        use_post_name += ("xt_" if use_tongxiao_post else "")
        use_post_name += ("qlh_" if use_post else "")
        use_post_name += ("_".join([str(i) for i in post_method]) if use_post else "")
    
        for model_idx, model in enumerate(self.models):
            # 收集输出结果
            predictions = []
            for idx, batch in enumerate(self.test_dataloader):
                output = model(batch, mode="ensemble", use_post=use_post, post_method=post_method, verbose=False)
                prediction = output["predictions"]
                predictions.append(prediction)
                
            # 输出格式转换            
            predictions_dict = dict()
            for k in predictions[0].keys():
                predictions_dict[k] = [pred for batch_pred in  predictions for pred in batch_pred[k]]
            prediction_results.append(predictions_dict)

            # 收集预估结果
            metric = model.span_f1.get_metric(True)
            
            record = {
                "model_name" : self.model_dirs[model_idx].split("/")[-3],
                "use_post_name": use_post_name,
            }
            print("="*100)
            print("="*100)
            print(f"test model {self.model_dirs[model_idx]}")

            for key in ["F1@CONST_DIR", "F1@LIMIT", "F1@OBJ_DIR", "F1@OBJ_NAME", "F1@PARAM", "F1@VAR", "micro@F1"]:
                if key in metric:
                    _key =  f"test_{key}"
                    record[_key] = f"{metric[key]:.5}"
                    print(f"{_key}:   {metric[key]:.5}")
            print("="*100)
            print("="*100)
            self.excel_metric_results.append(record)
            metric_results.append(metric)

        # 定义span标签
        gold_spans = prediction_results[0]["gold_spans"]
        for model_result in prediction_results:
            assert gold_spans[0] == model_result["gold_spans"][0]
        
        self.metric_results = metric_results
        self.excel_metric_results = pd.DataFrame.from_dict(self.excel_metric_results)

        if not online_submission_test:
            self.excel_metric_results.to_csv(f"{use_post_name}.csv")

        self.gold_spans = gold_spans
        self.prediction_results = prediction_results

    def get_ensemble_results(self, method=None):
        method = method if method is not None else self.method
        assert method in ["span-based", "tag-based", "logit-based"]

        if method == "span-based":
            mix_metric = self.ensemble_by_spans(self.gold_spans, self.prediction_results, self.metric_results)
        elif method == "tag-based":
            mix_metric = self.ensemble_by_tags(self.gold_spans, self.prediction_results)
        else: # method == "logit-based":
            mix_metric = self.ensemble_by_best_logits(self.gold_spans, self.prediction_results)

        print("*"*50)
        print("*"*50)
        print("method=", method)
        for key in ["F1@CONST_DIR", "F1@LIMIT", "F1@OBJ_DIR", "F1@OBJ_NAME", "F1@PARAM", "F1@VAR", "micro@F1"]:
            if key in mix_metric:
                _key =  f"test_{key}"
                print(f"{_key}:   {mix_metric[key]}")
        print("*"*50)
        print("*"*50)
        return mix_metric

    def ensemble_by_spans(self, gold_spans, prediction_results, metric_results):
        type_to_pred_idx = dict()
        for span_type in ["CONST_DIR", "LIMIT", "OBJ_DIR", "OBJ_NAME", "PARAM", "VAR"]:
            type_results = [metric["F1@" + span_type] for metric in metric_results]
            max_metric = max(type_results)
            type_to_pred_idx[span_type] = type_results.index( max_metric )

        mix_pred_spans = []
        for i in range( len(gold_spans) ):
            single_pred = dict()

            for span_type in ["CONST_DIR", "LIMIT", "OBJ_DIR", "OBJ_NAME", "PARAM", "VAR"]:
                single_prediction = prediction_results[ type_to_pred_idx[span_type] ]["pred_spans"][i]
                single_pred.update({
                    span_k: span_v for span_k, span_v in single_prediction.items() if span_v == span_type
                })
            
            mix_pred_spans.append( single_pred )
        
        self.mix_span_f1(mix_pred_spans, gold_spans)
        mix_metric = self.mix_span_f1.get_metric(True)
        return mix_metric
    
    def ensemble_by_tags(self, gold_spans, prediction_results):
        new_pred_tags = []
        for idx in range(len(gold_spans)):
            
            tags_merge = [[t] for t in prediction_results[0]["token_tags"][idx]]
            for model_idx, predictions_dict in enumerate(prediction_results):
                if model_idx == 0:
                    continue
                tags = predictions_dict["token_tags"][idx]
                for i, tag in enumerate(tags):
                    tags_merge[i] = tags_merge[i] + [tag]

            merge_tags = []
            for i, group in enumerate(tags_merge):
                tag, count = Counter(group).most_common(1)[0]
                merge_tags.append(tag)
        
            new_pred_tags.append(merge_tags)
        
        mix_pred_spans = []
        for mix_tags in new_pred_tags:
            mix_pred_spans.append(extract_spans(mix_tags, iob_tagging=self.models[0].iob_tagging))

        self.mix_span_f1(mix_pred_spans, gold_spans)
        mix_metric = self.mix_span_f1.get_metric(True)
        return mix_metric
    
    def ensemble_by_best_logits(self, gold_spans, prediction_results):
        new_pred_tags = []
        
        # (N*b, s, class)
        
        for idx in range(len(gold_spans)):
            
            logtis_merge = [t for t in prediction_results[0]["logits"][idx]]
            for model_idx, predictions_dict in enumerate(prediction_results):
                if model_idx == 0:
                    continue
                seq_logits = predictions_dict["logits"][idx]
                for i, logits in enumerate(seq_logits):
                    # sum the logit of each tag
                    logtis_merge[i] = logtis_merge[i] + logits
            
            merge_tags = []
            for i, logits in enumerate(logtis_merge):
                # get the max logit for the tag
                tag_idx = torch.argmax(logits).item()
                merge_tags.append(
                    self.models[0].id_to_tag[tag_idx])
               
            new_pred_tags.append(merge_tags)
        
        mix_pred_spans = []
        for mix_tags in new_pred_tags:
            mix_pred_spans.append(extract_spans(mix_tags, iob_tagging=self.models[0].iob_tagging))

        self.mix_span_f1(mix_pred_spans, gold_spans)
        mix_metric = self.mix_span_f1.get_metric(True)
        return mix_metric

if __name__ == '__main__':
    work_arg = parse_args_product()

    # x = occumpy_mem(os.environ["CUDA_VISIBLE_DEVICES"])

    "Need to use a new config file as common config"
    sg = PretrainedConfig.from_json_file(work_arg.config)
    print("="*50+"\n", vars(work_arg), "\n"+"="*50)
    print(sg)

    sg.use_post = work_arg.use_post if work_arg.use_post else False
    sg.use_tongxiao_post = work_arg.use_tongxiao_post if work_arg.use_tongxiao_post else False
    sg.use_label_standardize = work_arg.use_label_standardize if work_arg.use_label_standardize else False
    
    work_arg.post_method = [int(i) for i in work_arg.post_method]
    work_arg.label_method = [int(i) for i in work_arg.label_method]
    work_arg.gpu = f"cuda:{work_arg.gpu}"
    print("="*50+"\n", vars(work_arg), "\n"+"="*50)
    print(sg)

    pos_tag_to_idx = {} if not sg.pos_tag_dict else read_json(sg.pos_tag_dict)
    seed_everything(int(work_arg.seed))

    # load the dataset first
    test_dataset = get_reader(file_path=sg.test, target_vocab=get_tagset(sg.iob_tagging), encoder_model=sg.encoder_model, max_instances=sg.max_instances, max_length=sg.max_length,
                            
                            use_num_standardize=sg.use_num_standardize, use_pos_tag=sg.use_pos_tag, use_lemma=sg.use_lemma, pos_tag_to_idx=pos_tag_to_idx,
                            use_label_standardize=sg.use_label_standardize,
                            label_method=work_arg.label_method,
                            iob_tagging=sg.iob_tagging,

                            mode="test",
                            model_class = sg.model_class,
                            span_type= None if not hasattr(sg, 'span_type') else sg.span_type,
                            tongxiao_post=sg.use_tongxiao_post,
                            )

    test_dataloader = DataLoader(test_dataset,
                                # batch_size=sg.batch_size,
                                batch_size = 1,
                                collate_fn=test_dataset.collate_batch_fn, num_workers=12)

    # model_dirs = [
    #     "trained_model/baseline_aug_v3.2/xlmr_aug_v3.2_attack_fgm/version_2/checkpoints",
    #     "trained_model/baseline_aug_v3.2/xlmr_aug_v3.2_attack_fgm/version_3/checkpoints",
    #     "trained_model/baseline_aug_v3.2/xlmr_aug_v3.2_attack_fgm/version_4/checkpoints",
    #     "trained_model/baseline_aug_v3.2/xlmr_aug_v3.2_attack_pgd/version_1/checkpoints",
    # ]

    # model_dirs = [
    #     "trained_model/baseline_aug_v3.2/xlmr_aug_v3.2_attack_fgm_label_stand1_no_post/version_0/checkpoints",
    #     "trained_model/baseline_aug_v3.2/xlmr_aug_v3.2_attack_pgd_label_stand1_no_post/version_1/checkpoints",
    #     "trained_model/baseline_denoised_raw/xlmr_denoised_raw_no_stand_attack_fgm_label_stand1_no_post/version_0/checkpoints",
    #     "trained_model/baseline_denoised_raw/xlmr_denoised_raw_no_stand_attack_pgd_label_stand1_no_post/version_1/checkpoints",
    # ]
    # model_dirs = [
    #     "trained_model/baseline_aug_v3.2/xlmr_aug_v3.2_attack_fgm_remove_times/version_0/checkpoints",
    #     "trained_model/baseline_aug_v3.2/xlmr_aug_v3.2_attack_pgd_stand_remove_times/version_0/checkpoints",
    #     "trained_model/baseline_aug_v3.2/xlmr_aug_v3.2_attack_pgd_stand_remove_times2/version_0/checkpoints",
    # ]


    model_dirs = [
        "trained_model/baseline_aug_v3.2/xlmr_aug_v3.2_attack_fgm_remove_times/version_0/checkpoints",
        "trained_model/baseline_aug_v3.2/xlmr_aug_v3.2_attack_pgd_stand_remove_times2/version_0/checkpoints",

        "trained_model/baseline_aug_v3.2/xlmr_aug_v3.2_attack_pgd_remove_times2_len250/version_0/checkpoints",
        "trained_model/baseline_aug_v3.2/xlmr_large_aug_v3.2_attack_pgd_stand_remove_times/version_0/checkpoints",
        # "trained_model/baseline_aug_v3.2/xlmr_large_aug_v3.2_attack_pgd_stand_remove_times/version_2/checkpoints",
    ]

    """use new path for Repeting train.sh online"""
    if work_arg.online_submission_train:
        for i, model_dir in enumerate(model_dirs):
            model_dirs[i] = model_dir.replace("baseline_aug_v3.2", "baseline_aug_v3.2_new")
    
    # del x
    # torch.cuda.empty_cache()


    MODEL_CLASS = NerModel
    models = []
    for model_dir in model_dirs:
        model = MODEL_CLASS.from_pretrained(
            model_dir, gpu=work_arg.gpu
        )
        model.to(work_arg.gpu)
        models.append(model)
    
    predictor = IntegratorPredictor(
                model_dirs=model_dirs,
                models = models,
                test_dataloader=test_dataloader,
                gpu=work_arg.gpu
            )
    predictor.test(
        use_tongxiao_post=sg.use_tongxiao_post,
        use_post=sg.use_post,
        post_method=work_arg.post_method,
        online_submission_test=work_arg.online_submission_test
    )

    out = predictor.get_ensemble_results(method="span-based")

    # write the micro averaged F1 score to results.out
    write_for_leaderboard(out, "results.out")
    if not work_arg.online_submission_train:
        # to check
        write_for_leaderboard(out, "results_check1.out")
    else:
        # to check
        write_for_leaderboard(out, "results_check2.out")
