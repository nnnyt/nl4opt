import os
import torch

print("torch.cuda.is_available() = ", torch.cuda.is_available(), " |  device_count  ", torch.cuda.device_count())
# os.environ["CUDA_VISIBLE_DEVICES"]

from pytorch_lightning import seed_everything

from utils.utils import get_reader, parse_args_product, get_tagset, read_json, write_eval_performance
from transformers import PretrainedConfig
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup

from model.model import NerModel
from model.model_cascade import CascadeNerModel
from model.model_mrc import MRCNerModel
from model.model_fine import FineNerModel
from model.trainer import CustomTrainer

def write_for_leaderboard(out, out_filename):
    ''' write the micro averaged F1 score to results.out '''
    open(out_filename, 'wt').write(str(out["results"]["micro@F1"]))


if __name__ == '__main__':
    work_arg = parse_args_product()

    sg = PretrainedConfig.from_json_file(work_arg.config)
    print(sg)
    print("="*50+"\n", vars(work_arg), "\n"+"="*50)
    
    sg.test = work_arg.test if work_arg.test else sg.test
    sg.use_post = work_arg.use_post if work_arg.use_post else False
    sg.use_tongxiao_post = work_arg.use_tongxiao_post if work_arg.use_tongxiao_post else False
    sg.use_label_standardize = work_arg.use_label_standardize if work_arg.use_label_standardize else False
    
    work_arg.post_method = [int(i) for i in work_arg.post_method]
    work_arg.label_method = [int(i) for i in work_arg.label_method]
    work_arg.gpu = f"cuda:{work_arg.gpu}"
    print(sg)
    print("="*50+"\n", vars(work_arg), "\n"+"="*50)

    pos_tag_to_idx = {} if not sg.pos_tag_dict else read_json(sg.pos_tag_dict)
    seed_everything(int(work_arg.seed))
    
    # load the dataset first
    test_dataset = get_reader(file_path=sg.test, target_vocab=get_tagset(sg.iob_tagging), encoder_model=sg.encoder_model, max_instances=sg.max_instances, max_length=sg.max_length,
                            
                            use_num_standardize=sg.use_num_standardize, use_pos_tag=sg.use_pos_tag, use_lemma=sg.use_lemma, pos_tag_to_idx=pos_tag_to_idx,
                            use_label_standardize=sg.use_label_standardize,
                            label_method = work_arg.label_method,
                            iob_tagging=sg.iob_tagging,

                            mode="test",
                            model_class = sg.model_class,
                            span_type= None if not hasattr(sg, 'span_type') else sg.span_type,
                            tongxiao_post=sg.use_tongxiao_post
                            )

    test_dataloader = DataLoader(test_dataset,
                                # batch_size=sg.batch_size,s
                                batch_size = 1,
                                collate_fn=test_dataset.collate_batch_fn, num_workers=12)

    model_dir = work_arg.model + "/checkpoints"
    if sg.model_class == "cascade":
        MODEL_CLASS = CascadeNerModel
    if sg.model_class == "mrc":
        MODEL_CLASS = MRCNerModel
    elif sg.model_class == "fine":
        MODEL_CLASS = FineNerModel
        # triggers = ["trigger_param", "trigger_obj_dir", "trigger_limit"]
        # tokenizer = test_dataset.tokenizer
        # trigger_to_idx = {trigger: tokenizer.vocab[trigger] for trigger in triggers}
        # print("DEBUG: trigger_to_idx=", trigger_to_idx)
        # pretrained_model_path = sg.pretrained_model_path
    else:
        MODEL_CLASS = NerModel
    model = MODEL_CLASS.from_pretrained(
                            model_dir,
                            gpu=work_arg.gpu
                        )
    model.to(work_arg.gpu)
    
    trainer = CustomTrainer(
                model = model,
                test_dataloader=test_dataloader,
            )
    # when submit for online testing, stop verbose, or you would get error !
    if not work_arg.online_submission_test:
        out = trainer.test(use_post=sg.use_post, post_method = work_arg.post_method, verbose=True)
        
        print("test no post process ....> ")
        trainer.test(use_post=False, verbose=False)
        print("***************** Finish Testing for debug*****************")
    else:
        out = trainer.test(use_post=sg.use_post, post_method = work_arg.post_method, verbose=False)
    
    # write the micro averaged F1 score to results.out
    write_for_leaderboard(out, "results.out")