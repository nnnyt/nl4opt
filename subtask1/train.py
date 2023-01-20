import os
from pickle import TRUE
import torch

print("torch.cuda.is_available() = ", torch.cuda.is_available(), " |  device_count  ", torch.cuda.device_count())
# os.environ["CUDA_VISIBLE_DEVICES"]

from pytorch_lightning import seed_everything
# seed_everything(42)

from utils.utils import get_reader, parse_args_product, get_tagset, read_json, occumpy_mem
from transformers import PretrainedConfig
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup
from model.model import NerModel

from model.model_cascade import CascadeNerModel
from model.model_mrc import MRCNerModel
from model.model_fine import FineNerModel
from model.train_utils import build_optimizer
from model.trainer import CustomTrainer

if __name__ == '__main__':

    work_arg = parse_args_product()

    # x = occumpy_mem(work_arg.gpu)

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
                            label_method = work_arg.label_method,
                            iob_tagging=sg.iob_tagging,
                            
                            mode="train",
                            model_class = sg.model_class,
                            span_type= None if not hasattr(sg, 'span_type') else sg.span_type,
                            tongxiao_post=sg.use_tongxiao_post,
                            
                            use_dynamic_mask=False if not hasattr(sg, 'use_dynamic_mask') else sg.use_dynamic_mask,
                            )
    # 标签标准化：与训练集一致，目的是保证标准化的标签和训练集的分布相近，从而保存的最好的模型 不受这部分标签的影响（如果真的分布不同，则通过后处理解决）
    valid_dataset = get_reader(file_path=sg.dev, target_vocab=get_tagset(sg.iob_tagging), encoder_model=sg.encoder_model, max_instances=sg.max_instances, max_length=sg.max_length,
                            
                            use_num_standardize=sg.use_num_standardize, use_pos_tag=sg.use_pos_tag, use_lemma=sg.use_lemma, pos_tag_to_idx=pos_tag_to_idx,
                            use_label_standardize=sg.use_label_standardize,
                            label_method = work_arg.label_method,
                            iob_tagging=sg.iob_tagging,

                            mode="valid",
                            model_class = sg.model_class,
                            span_type= None if not hasattr(sg, 'span_type') else sg.span_type,
                            tongxiao_post=True
                            )
    test_dataset = get_reader(file_path=sg.test, target_vocab=get_tagset(sg.iob_tagging), encoder_model=sg.encoder_model, max_instances=sg.max_instances, max_length=sg.max_length,
                            
                            use_num_standardize=sg.use_num_standardize, use_pos_tag=sg.use_pos_tag, use_lemma=sg.use_lemma, pos_tag_to_idx=pos_tag_to_idx,
                            use_label_standardize=False,
                            label_method = [],
                            iob_tagging=sg.iob_tagging,

                            mode="test",
                            model_class = sg.model_class,
                            span_type= None if not hasattr(sg, 'span_type') else sg.span_type,
                            tongxiao_post=True
                            )
    # del x
    # torch.cuda.empty_cache()

    train_dataloader = DataLoader(train_dataset, batch_size=sg.batch_size, collate_fn=train_dataset.collate_batch_fn, num_workers=12)
    valid_dataloader = DataLoader(valid_dataset, batch_size=sg.batch_size, collate_fn=valid_dataset.collate_batch_fn, num_workers=12)
    # train_dataloader = None
    # valid_dataloader = None
    
    test_dataloader = DataLoader(test_dataset, batch_size=sg.batch_size, collate_fn=test_dataset.collate_batch_fn, num_workers=12)

    trigger_to_idx = [] # 用于 fine_model
    pretrained_model_path = None
    if sg.model_class == "cascade":
        MODEL_CLASS = CascadeNerModel
        print("[DEBUG] " , sg.model_class)
    elif sg.model_class == "mrc":
        MODEL_CLASS = MRCNerModel
    elif sg.model_class == "fine":
        MODEL_CLASS = FineNerModel
        triggers = ["trigger_param", "trigger_obj_dir", "trigger_limit"]
        tokenizer = test_dataset.tokenizer
        trigger_to_idx = {trigger: tokenizer.vocab[trigger] for trigger in triggers}
        print("DEBUG: trigger_to_idx=", trigger_to_idx)
        pretrained_model_path = sg.pretrained_model_path
    else:
        MODEL_CLASS = NerModel
    model = MODEL_CLASS(dropout_rate=0.1,
                    tag_to_id=get_tagset(sg.iob_tagging),
                    iob_tagging=sg.iob_tagging,
                    encoder_model=sg.encoder_model,

                    use_pos_tag=sg.use_pos_tag, 
                    pos_tag_to_idx=pos_tag_to_idx,
                    pos_tag_coprus_size=len(pos_tag_to_idx),
                    pos_tag_embed_dim=sg.pos_tag_embed_dim,
                    pos_tag_pad_idx=0,

                    use_position=False if not hasattr(sg, 'use_position') else sg.use_position,

                    use_crf=True if not hasattr(sg, 'use_crf') else sg.use_crf,
                    use_cls=None if not hasattr(sg, 'use_cls') else sg.use_cls,
                    use_wof=False if not hasattr(sg, 'use_wof') else sg.use_wof,

                    trigger_to_idx=trigger_to_idx,
                    pretrained_model_path=pretrained_model_path,
                    use_lstm=False if not hasattr(sg, 'use_lstm') else sg.use_lstm,
                    gpu=work_arg.gpu,
                )
    model.to(work_arg.gpu)
    if False:
        # 统一学习率
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(params, lr=sg.lr, weight_decay=0.01)
    else:
        # 差分学习率
        # optimizer = build_optimizer(model, weight_decay=0.01, lr=1e-5, crf_lr=1e-2, other_lr=1e-4)
        # 
        optimizer = build_optimizer(model, weight_decay=0.01, lr=sg.lr, crf_lr=sg.lr*1000, other_lr=sg.lr*10)
        # --lr=3e-5 \
        # --crf_lr=3e-2 \
        # --other_lr=3e-4 \
    
    if False:
        # 自适应 递减学习率
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.25, patience=5, verbose=True, min_lr=1e-7)
    else:
        # Warmingpu + 递减学习率
        train_batches = len(train_dataset) // (sg.batch_size * 1)
        decreasing_lr=work_arg.decreasing_lr,
        if decreasing_lr:
            total_steps =  51 * train_batches # old: 50 25
        else:
            total_steps = (sg.epochs * 50) * train_batches # old: 50 25
        
        warmup_steps = int(train_batches * 0.5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    trainer = CustomTrainer(
                model = model,
                train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, test_dataloader=test_dataloader,
                
                out_model_dir=out_dir_path, 
                out_model_name=sg.model_name,

                epochs=sg.epochs,
                grad_accum=sg.accum_grad_batches,

                use_attack_method = None  if not hasattr(sg, 'use_attack_method') else sg.use_attack_method,
            )
    trainer.setup_env(optimizer, scheduler)
    trainer.train(use_post=sg.use_post, post_method=work_arg.post_method)
    print("***************** Finish Training *****************")
    print()
    print()
    
    best_model = MODEL_CLASS.from_pretrained(
                    trainer.bcp_path,
                    gpu=work_arg.gpu
                )
    best_model.to(work_arg.gpu)
    trainer.valid(best_model, use_post=True, post_method=work_arg.post_method)
    print("valid with use_post=True <--")
    print("***************** Finish Validation *****************")
    print()
    print()

    trainer.test(best_model, test_dataloader, use_post=True, post_method=work_arg.post_method, verbose=False)
    print("test with use_post=True <--")
    print()
    # trainer.test(best_model, test_dataloader, use_post=True, post_method=work_arg.post_method, verbose=False)
    # print("test with use_post=True <--")
    # print("***************** Finish Testing *****************")
