import argparse
import os
import time

import torch
from pytorch_lightning import seed_everything

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from log import logger
from model.ner_model import NERBaseAnnotator
from model.ner_model_w_feat import NERBaseAnnotator_W_FEAT
from utils.reader import CoNLLReader, CascadeCoNLLReader, MrcCoNLLReader

import json
def read_json(open_path):
    load_data = json.load(open(open_path, 'r', encoding='utf-8'))
    print("[read_json] num = {}, open_path = {}".format(len(load_data), open_path))
    return load_data

def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    print(f"Chcek CUDA: total={total}, used={used}")
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = min(max_mem - used, 3000)
    print(f"use block_mem = {block_mem}")

    # 1M * bock_num
    x = torch.cuda.FloatTensor(1*256, 1024, block_mem)
    return x


conll_iob = {
    'B-LIMIT':0,
    'I-LIMIT':1,
    'B-CONST_DIR':2,
    'I-CONST_DIR':3,
    'B-VAR':4,
    'I-VAR':5,
    'B-PARAM':6,
    'I-PARAM':7,
    'B-OBJ_NAME':8,
    'I-OBJ_NAME':9,
    'B-OBJ_DIR':10,
    'I-OBJ_DIR':11,
    'O':12
}

conll_ob = {
    'B-LIMIT':0,
    'B-CONST_DIR':1,
    'B-VAR':2,
    'B-PARAM':3,
    'B-OBJ_NAME':4,
    'B-OBJ_DIR':5,
    'O':6
}


def parse_args_product():
    p = argparse.ArgumentParser(description='Model configuration.', add_help=False)

    p.add_argument('--config', type=str, help='model config file path', default=None)

    p.add_argument('--model', type=str, help='Model path.', default=None) # 用于finetune 或 evaluate

    p.add_argument('--lr', type=float, help='Learning rate', default=1e-6) # 用于finetune
    p.add_argument('--epochs', type=int, help='Number of epochs for training.', default=25) # 用于finetune


    p.add_argument('--train', type=str, help='Model path.', default=None) # 用于fold
    p.add_argument('--dev', type=str, help='Model path.', default=None) # 用于fold
    p.add_argument('--test', type=str, help='Model path.', default=None) # 用于fold

    p.add_argument('--use_post', action="store_true", default=None) # 用于后处理
    p.add_argument('--post_method', type=str, help='methods for use_post', default="125") # test1: 125, test2: 12 3
    p.add_argument('--use_tongxiao_post', action="store_true", default=None) # 用于后处理

    p.add_argument('--use_label_standardize', action="store_true", default=None) # 用于后处理
    p.add_argument('--label_method', type=str, help='methods for use_label_standardize', default="0") # test1: 01, test2: 01 3

    p.add_argument('--gpu', type=str, help='cuda idx', default='0')
    p.add_argument('--decreasing_lr', action="store_true", default=None)

    p.add_argument('--online_submission_test', action="store_true", default=None)
    p.add_argument('--online_submission_train', action="store_true", default=None)
    p.add_argument('--seed', type=int, help='seed', default=45) # use for test and integrator
    return p.parse_args()


def get_tagset(tagging_scheme):

    if "conll_ob" in tagging_scheme:
        return conll_ob
    elif tagging_scheme == "bio_const_dir":
        return {
            'B-CONST_DIR':0,
            'I-CONST_DIR':1,
            'O': 2
        }
    elif tagging_scheme == "bio_obj_name":
        return {
            'B-OBJ_NAME':0,
            'I-OBJ_NAME':1,
            'O': 2
        }
    elif tagging_scheme == "bio_var":
        return {
            'B-VAR':0,
            'I-VAR':1,
            'O': 2
        }
    else:
        return conll_iob
    # else:
    #     # If you choose to use a different tagging scheme, you may need to do some post-processing
    #     raise Exception("ERROR: Only conll tagging scheme is accepted")


def get_out_filename(out_dir, model, prefix):
    model_name = os.path.basename(model)
    model_name = model_name[:model_name.rfind('.')]
    return '{}/{}_base_{}.tsv'.format(out_dir, prefix, model_name)


def write_eval_performance(eval_performance, out_file):
    outstr = ''
    added_keys = set()
    for out_ in eval_performance:
        for k in out_:
            if k in added_keys or k in ['results', 'predictions']:
                continue
            outstr = outstr + '{}\t{}\n'.format(k, out_[k])
            added_keys.add(k)

    open(out_file, 'wt').write(outstr)
    logger.info('Finished writing evaluation performance for {}'.format(out_file))


def get_reader(file_path, max_instances=-1, max_length=50, target_vocab=None, encoder_model='xlm-roberta-base',
                use_num_standardize=False, use_pos_tag=False, use_lemma=False, pos_tag_to_idx={},
                use_label_standardize=True,
                label_method=[0],
                iob_tagging="conll",
                mode="train",
                model_class = "",
                use_dynamic_mask=False,
                span_type=None,
                tongxiao_post=False,
                ):
    if file_path is None:
        return None
    if model_class == "cascade":
        READER_CLASS = CascadeCoNLLReader
    elif model_class == "mrc":
        READER_CLASS = MrcCoNLLReader
    else:
        READER_CLASS = CoNLLReader
    reader = READER_CLASS(max_instances=max_instances, max_length=max_length, target_vocab=target_vocab, encoder_model=encoder_model,
                        
                        use_num_standardize=use_num_standardize,
                        use_label_standardize=use_label_standardize,
                        label_method=label_method,

                        use_pos_tag=use_pos_tag,
                        use_lemma=use_lemma,
                        pos_tag_to_idx=pos_tag_to_idx,

                        iob_tagging=iob_tagging,
                        tag_to_id=get_tagset(iob_tagging),
                        
                        mode=mode,
                        use_dynamic_mask=use_dynamic_mask,

                        span_type=span_type,
                        tongxiao_post=tongxiao_post,
                        )
    if model_class == "fine":
        # 添加 trigger 到词典
        triggers = ["trigger_param", "trigger_obj_dir", "trigger_limit"]
        for trigger in triggers:
            if trigger in reader.tokenizer.vocab:
                continue
            print(f"[get_reader] set triggers for tokenizer ! ({trigger})")
            reader.tokenizer.add_special_tokens({
                'additional_special_tokens': [trigger]
            })

    reader.read_data(file_path)

    return reader


def create_model(train_data, dev_data, tag_to_id, batch_size=64, dropout_rate=0.1,
                stage='fit',
                iob_tagging='conll',
                lr=1e-5, encoder_model='xlm-roberta-base', num_gpus=1,
                model_class="default", 
                use_pos_tag=False, pos_tag_to_idx={}, pos_tag_coprus_size=None, pos_tag_embed_dim=None,
                
                use_position=False,
                use_crf=True,
                use_cls=None,
                use_wof=False,
                ):
    if model_class == "default":
        MODEL_CLASS = NERBaseAnnotator
    elif model_class == "enhance_feat":
        MODEL_CLASS = NERBaseAnnotator_W_FEAT
    else:
        raise ValueError("Illegal MODEL !")
    
    return MODEL_CLASS(train_data=train_data, dev_data=dev_data, tag_to_id=tag_to_id, batch_size=batch_size,
                            stage=stage,
                            iob_tagging=iob_tagging,
                            encoder_model=encoder_model,
                            dropout_rate=dropout_rate, lr=lr, pad_token_id=train_data.pad_token_id, num_gpus=num_gpus,
                            
                            use_pos_tag=use_pos_tag,
                            pos_tag_to_idx=pos_tag_to_idx,
                            pos_tag_coprus_size=pos_tag_coprus_size,
                            pos_tag_embed_dim=pos_tag_embed_dim,

                            pos_tag_pad_idx=pos_tag_to_idx.get("PAD", 0),

                            use_position=use_position,

                            use_crf=use_crf,
                            use_cls=use_cls,
                            use_wof=use_wof
                            )

def load_model(model_file, tag_to_id=None,
                stage='test', iob_tagging='conll',
                lr=1e-5, batch_size=16, num_gpus=1,
                model_class="default",
               ):
    if ~os.path.isfile(model_file):
        model_file = get_models_for_evaluation(model_file)

    print("[debug] load_model , model_file=", model_file)
    hparams_file = model_file[:model_file.rindex('checkpoints/')] + '/hparams.yaml'
    """使用默认学习率"""
    if model_class == "default":
        MODEL_CLASS = NERBaseAnnotator
    elif model_class == "enhance_feat":
        MODEL_CLASS = NERBaseAnnotator_W_FEAT
    model = MODEL_CLASS.load_from_checkpoint(model_file, hparams_file=hparams_file, stage=stage, tag_to_id=tag_to_id, batch_size=batch_size, num_gpus=num_gpus,
                                            iob_tagging=iob_tagging)
    model.stage = stage
    return model, model_file


def save_model(trainer, out_dir, model_name, timestamp=None):
    out_dir = out_dir + '/lightning_logs/version_' + str(trainer.logger.version) + '/checkpoints/'
    if timestamp is None:
        timestamp = time.time()
    os.makedirs(out_dir, exist_ok=True)

    outfile = out_dir + '/' + model_name + '_timestamp_' + str(timestamp) + '_final.ckpt'
    trainer.save_checkpoint(outfile, weights_only=True)

    logger.info('Stored model {}.'.format(outfile))
    return outfile


def train_model(model, out_dir='', epochs=10, gpus=1, model_name='', timestamp='', grad_accum=1):
    trainer = get_trainer(gpus=gpus, out_dir=out_dir, epochs=epochs, model_name=model_name, timestamp=timestamp, grad_accum=grad_accum)
    trainer.fit(model)
    print("Finish training .... ")

    return trainer


def get_modelcheckpoint_callback(out_dir, model_name, timestamp):
    if not os.path.exists(out_dir + '/lightning_logs/'):
        os.makedirs(out_dir + '/lightning_logs/')

    if len(os.listdir(out_dir + '/lightning_logs/')) < 1:
        bcp_path  = out_dir + '/lightning_logs/version_0/checkpoints/'
    else:
        bcp_path  = out_dir + '/lightning_logs/version_' +  str(int(os.listdir(out_dir + '/lightning_logs/')[-1].split('_')[-1]) + 1) + '/checkpoints/'
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_micro@F1',
        mode='max',
        save_last=False,
        dirpath=bcp_path,
        filename=model_name + '_timestamp_' + str(timestamp) + '_{epoch:02d}_{val_micro@F1:.4f}_best'
    )
    return checkpoint_callback


def get_trainer(gpus=4, is_test=False, out_dir=None, epochs=10, model_name='', timestamp='', grad_accum=1):
    seed_everything(42)
    # logger = pl.loggers.CSVLogger(out_dir, name="lightning_logs")
    logger = pl.loggers.TensorBoardLogger(out_dir, name='lightning_logs')

    if is_test:
        return pl.Trainer(gpus=1, logger=logger) if torch.cuda.is_available() else pl.Trainer(val_check_interval=100)

    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=gpus, logger=logger, deterministic=True, max_epochs=epochs, callbacks=[get_model_earlystopping_callback(), get_modelcheckpoint_callback(out_dir, model_name, timestamp)],
                        default_root_dir=out_dir, distributed_backend='ddp', checkpoint_callback=True, accumulate_grad_batches=grad_accum,
                        # num_sanity_val_steps=1,
                        )
        trainer.callbacks.append(get_lr_logger())
    else:
        trainer = pl.Trainer(max_epochs=epochs, logger=logger, default_root_dir=out_dir)

    return trainer


def get_lr_logger():
    lr_monitor = LearningRateMonitor(logging_interval='step')
    return lr_monitor


def get_model_earlystopping_callback():
    es_clb = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10, # old: 5
        verbose=True,
        mode='min'
    )
    return es_clb


def get_models_for_evaluation(path):
    if 'checkpoints' not in path:
        path = path + '/checkpoints/'
    model_files = list_files(path)
    models = [f for f in model_files if f.endswith('best.ckpt')]

    return models[0] if len(models) != 0 else None


def list_files(in_dir):
    files = []
    for r, d, f in os.walk(in_dir):
        for file in f:
            files.append(os.path.join(r, file))
    return files
