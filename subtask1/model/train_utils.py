import os
import time
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
import torch


def load_wise_model(model, model_state_path, single_gpu=True, gpu="cuda:0"):

    if single_gpu: # 单GPU
      new_state_dict = {k[:7].replace('module.', '')+k[7:] : v for k, v in torch.load(model_state_path, map_location=torch.device(gpu)).items()}
    
    else: # 多GPU
      new_state_dict = OrderedDict()
      # 修改 key，没有module字段保持，如果有，则需要删除 module.
      for k, v in torch.load(model_state_path).items():
          if 'module.' in k[:7]:
              k = k
          else:
              k = "module." + k
          new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(torch.load(output_model_file)) #, map_location='cpu'))
    return model

def save_wise_model(model, model_state_path, single_gpu=True):
    save_dict = model.state_dict() if single_gpu else model.module.state_dict()
    torch.save(save_dict, model_state_path)



def set_model_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if len(os.listdir(out_dir)) < 1:
        bcp_path  = out_dir + '/version_0/checkpoints/'
    else:
        bcp_path  = out_dir + '/version_' +  str(int(os.listdir(out_dir)[-1].split('_')[-1]) + 1) + '/checkpoints/'
    
    return bcp_path

def list_files(in_dir):
    files = []
    for r, d, f in os.walk(in_dir):
        for file in f:
            files.append(os.path.join(r, file))
    return files

def get_pretrained_model_path(path):
    if 'checkpoints' not in path:
        path = path + '/checkpoints/'
    model_files = list_files(path)
    models = [f for f in model_files if f.endswith('best.bin')]

    return models[0] if len(models) != 0 else None



def build_optimizer(model, weight_decay=0.01, lr=1e-5, crf_lr=1e-5, other_lr=1e-5):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    crf_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        # 过滤 不求 梯度 的参数
        # if not para.requires_grad:
        #     continue
        
        space = name.split('.')
        # print(name)
        if space[0] == 'bert' or space[0] == 'encoder':
            bert_param_optimizer.append((name, para))
        elif 'crf_layer' in space[0]:
            crf_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay, 'lr': lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': lr},

        # crf模块
        {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay, 'lr': crf_lr},
        {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': other_lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay, 'lr': other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': other_lr},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer




