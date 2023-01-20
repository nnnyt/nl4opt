import os
import torch

print("torch.cuda.is_available() = ", torch.cuda.is_available(), " |  device_count  ", torch.cuda.device_count())
# os.environ["CUDA_VISIBLE_DEVICES"]

from pytorch_lightning import seed_everything
# seed_everything(42)

from utils.utils import get_reader, parse_args_product, get_tagset, read_json
from transformers import PretrainedConfig
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup

from model.model import NerModel
from model.model_cascade import CascadeNerModel
from model.train_utils import build_optimizer
from model.trainer import CustomTrainer

if __name__ == '__main__':
    work_arg = parse_args_product()

    sg = PretrainedConfig.from_json_file(work_arg.config)
    print(sg)
    print("="*50+"\n", vars(work_arg), "\n"+"="*50)
    sg.train = work_arg.train if work_arg.train else sg.train
    sg.dev = work_arg.dev if work_arg.dev else sg.dev
    sg.test = work_arg.test if work_arg.test else sg.test
    sg.use_post = work_arg.use_post if work_arg.use_post else sg.use_post
    sg.use_tongxiao_post = work_arg.use_tongxiao_post if work_arg.use_tongxiao_post else False
    sg.use_label_standardize = work_arg.use_label_standardize if work_arg.use_label_standardize else sg.use_label_standardize

    work_arg.post_method = [1,2,5] # [int(i) for i in work_arg.post_method]
    work_arg.label_method = [0] # [int(i) for i in work_arg.label_method]
    work_arg.gpu = f"cuda:{work_arg.gpu}"
    print(sg)
    print("="*50+"\n", vars(work_arg), "\n"+"="*50)

    """use new path for Repeting train.sh online"""
    if work_arg.online_submission_train:
        out_dir_path = sg.out_dir + '_new/' + sg.model_name
    else:
        out_dir_path = sg.out_dir + '/' + sg.model_name

    pos_tag_to_idx = {} if not sg.pos_tag_dict else read_json(sg.pos_tag_dict)
    if int(work_arg.seed) > 0:
        seed_everything(int(work_arg.seed))
    
    # load the dataset first
    train_dataset = get_reader(file_path=sg.train, target_vocab=get_tagset(sg.iob_tagging), encoder_model=sg.encoder_model, max_instances=sg.max_instances, max_length=sg.max_length,
                            
                            use_num_standardize=sg.use_num_standardize, use_pos_tag=sg.use_pos_tag, use_lemma=sg.use_lemma, pos_tag_to_idx=pos_tag_to_idx,
                            use_label_standardize=sg.use_label_standardize,
                            iob_tagging=sg.iob_tagging,
                            
                            mode="train",
                            model_class = sg.model_class,
                            span_type= None if not hasattr(sg, 'span_type') else sg.span_type,
                            tongxiao_post=sg.use_tongxiao_post,

                            use_dynamic_mask=False if not hasattr(sg, 'use_dynamic_mask') else sg.use_dynamic_mask,
                            )
    valid_dataset = get_reader(file_path=sg.dev, target_vocab=get_tagset(sg.iob_tagging), encoder_model=sg.encoder_model, max_instances=sg.max_instances, max_length=sg.max_length,
                            
                            use_num_standardize=sg.use_num_standardize, use_pos_tag=sg.use_pos_tag, use_lemma=sg.use_lemma, pos_tag_to_idx=pos_tag_to_idx,
                            use_label_standardize=sg.use_label_standardize,
                            iob_tagging=sg.iob_tagging,

                            mode="valid",
                            model_class = sg.model_class,
                            span_type= None if not hasattr(sg, 'span_type') else sg.span_type,
                            tongxiao_post=True,
                            )
    test_dataset = get_reader(file_path=sg.test, target_vocab=get_tagset(sg.iob_tagging), encoder_model=sg.encoder_model, max_instances=sg.max_instances, max_length=sg.max_length,
                            
                            use_num_standardize=sg.use_num_standardize, use_pos_tag=sg.use_pos_tag, use_lemma=sg.use_lemma, pos_tag_to_idx=pos_tag_to_idx,
                            use_label_standardize=False,
                            iob_tagging=sg.iob_tagging,

                            mode="test",
                            model_class = sg.model_class,
                            span_type= None if not hasattr(sg, 'span_type') else sg.span_type,
                            tongxiao_post=True,
                            )

    train_dataloader = DataLoader(train_dataset, batch_size=sg.batch_size, collate_fn=train_dataset.collate_batch_fn, num_workers=12)
    valid_dataloader = DataLoader(valid_dataset, batch_size=sg.batch_size, collate_fn=valid_dataset.collate_batch_fn, num_workers=12)

    test_dataloader = DataLoader(test_dataset, batch_size=sg.batch_size, collate_fn=test_dataset.collate_batch_fn, num_workers=12)


    model_dir = work_arg.model + "/checkpoints"
    if sg.model_class == "cascade":
        MODEL_CLASS = CascadeNerModel
    else:
        MODEL_CLASS = NerModel
    model = MODEL_CLASS.from_pretrained(
                            model_dir,
                            gpu=work_arg.gpu
                        )
    model.to(work_arg.gpu)

    if False:
        optimizer = torch.optim.AdamW(model.parameters(), lr=work_arg.lr, weight_decay=0.01)
    else:
        # optimizer = build_optimizer(model, weight_decay=0.01, lr=1e-5, crf_lr=1e-2, other_lr=1e-4)
        optimizer = build_optimizer(model, weight_decay=0.01, lr=work_arg.lr, crf_lr=sg.lr*1000, other_lr=sg.lr*10)
        # --lr=3e-5 \
        # --crf_lr=3e-2 \
        # --other_lr=3e-4 \
    
    if False:
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.25, patience=5, verbose=True, min_lr=1e-7)
    else:
        train_batches = len(train_dataset) // (sg.batch_size * 1)
        total_steps = (work_arg.epochs * 50) * train_batches # old: 50 25
        warmup_steps = int(train_batches * 0.5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    trainer = CustomTrainer(
                model = model,
                train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, test_dataloader=test_dataloader,
                
                out_model_dir=out_dir_path, 
                out_model_name=sg.model_name,

                epochs=work_arg.epochs,
                grad_accum=sg.accum_grad_batches,

                use_attack_method = None  if not hasattr(sg, 'use_attack_method') else sg.use_attack_method,
            )
    trainer.setup_env(optimizer, scheduler)
    trainer.train(use_post=sg.use_post)
    print("***************** Finish Funing *****************")
    print()
    print()
    
    best_model = MODEL_CLASS.from_pretrained(
                    trainer.bcp_path,
                    gpu=work_arg.gpu
                )
    best_model.to(work_arg.gpu)
    trainer.valid(best_model, use_post=True)
    print("valid with use_post=True <--")
    print("***************** Finish Validation *****************")
    print()
    print()

    trainer.test(best_model, test_dataloader, use_post=True, verbose=False)
    print("test with use_post=True <--")
    print()
    # trainer.test(best_model, test_dataloader, use_post=True, verbose=False)
    # print("test with use_post=True <--")
    # print("***************** Finish Testing *****************")
