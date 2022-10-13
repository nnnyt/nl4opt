import os
from argparse import ArgumentParser

import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
#from config import Config
from config_attack import Config_attack
from config_span import Config_span
#from model import TextMappingModel
from model_span import TextMappingModel_span
from model_attack import TextMappingModel_attack
from data import LPMappingDataset
#from data_per_declaration import DeclarationMappingDataset
from data_per_declaration_span import DeclarationMappingDataset_span
from data_per_declaration_attack import DeclarationMappingDataset_attack
from constants import SPECIAL_TOKENS
from utils import *
# from rouge import Rouge
import test_utils_mix as test_utils

############

# configuration
parser = ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--checkpoint-c', type=str, required=True)
# parser.add_argument('--eval-train', action="store_true", dest="eval_train")
# parser.add_argument('--no-eval-train', action="store_false", dest="eval_train")
parser.add_argument('--test-file', type=str, default="")
# parser.add_argument('--debug', action="store_true", dest="debug")
# parser.add_argument('--no-debug', action="store_false", dest="debug")

# parser.set_defaults(eval_train=False)
parser.set_defaults(debug=False)

args = parser.parse_args()

use_gpu = args.gpu > -1
checkpoint = torch.load(args.checkpoint, map_location=f'cuda:{args.gpu}' if use_gpu else 'cpu') 
checkpoint_c = torch.load(args.checkpoint_c, map_location=f'cuda:{args.gpu}' if use_gpu else 'cpu')
config = Config_span.from_dict(checkpoint['config']) 
config_c = Config_attack.from_dict(checkpoint_c['config'])

# Override test file
if args.test_file:
    config.test_file = args.test_file
    config_c.test_file = args.test_file 

# set GPU device
config.gpu_device = args.gpu
config.use_gpu = use_gpu
# fix random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

print(config)

# set GPU device
config_c.gpu_device = args.gpu
config_c.use_gpu = use_gpu
# fix random seed
random.seed(config_c.seed)
np.random.seed(config_c.seed)
torch.manual_seed(config_c.seed)
torch.backends.cudnn.enabled = False

if use_gpu and config_c.gpu_device >= 0:
    torch.cuda.set_device(config_c.gpu_device)

print(config_c)

# datasets
model_name = config.bert_model_name

model_name_c = config_c.bert_model_name   

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          cache_dir=config.bert_cache_dir)

tokenizer_c = AutoTokenizer.from_pretrained(model_name_c,
                                          cache_dir=config_c.bert_cache_dir)
#############
tokenizer.add_tokens(SPECIAL_TOKENS)
tokenizer_c.add_tokens(SPECIAL_TOKENS) ############

# Output result files
ckpt_path_splits = args.checkpoint.split('/')
ckpt_basename = ckpt_path_splits[-1].split('.')[0]
if args.debug:
    ckpt_basename = 'debug-' + ckpt_basename

output_test_prefix = os.path.basename(args.test_file).split('.')[0] if args.test_file else "test"
output_dir = '/'.join(ckpt_path_splits[:-1])
test_result_file = os.path.join(output_dir, 'results', ckpt_basename, '{}.out.json'.format(output_test_prefix))
test_score_file = os.path.join(output_dir, 'results', ckpt_basename, '{}.score.json'.format(output_test_prefix))

os.makedirs(os.path.join(output_dir, 'results', ckpt_basename), exist_ok=True)

train_set = None

print('============== Prepare Test Set: starting =================')
vocabs = {}
test_set = DeclarationMappingDataset_span(config.test_file, max_length=config.max_length, gpu=use_gpu)
test_set.numberize(tokenizer, vocabs)
test_set_c = DeclarationMappingDataset_attack(config_c.test_file, max_length=config_c.max_length, gpu=use_gpu)
test_set_c.numberize(tokenizer_c, vocabs)  ###############
print('============== Prepare Test Set: finished =================')


# initialize the model
model = TextMappingModel_span(config, vocabs)
model.load_bert(model_name, cache_dir=config.bert_cache_dir, tokenizer=tokenizer)
if not model_name.startswith('roberta'):
    model.bert.resize_token_embeddings(len(tokenizer))
model.load_state_dict(checkpoint['model'], strict=True)

model_c = TextMappingModel_attack(config_c, vocabs)        ############
model_c.load_bert(model_name_c, cache_dir=config_c.bert_cache_dir, tokenizer=tokenizer_c)
if not model_name_c.startswith('roberta'):
    model_c.bert.resize_token_embeddings(len(tokenizer_c))
model_c.load_state_dict(checkpoint_c['model'], strict=True)

if use_gpu:
    model.cuda(device=config.gpu_device)
    model_c.cuda(device=config_c.gpu_device)     ###############
epoch = 1000

# Number of batches
batch_num = len(test_set) // config.eval_batch_size + \
                (len(test_set) % config.eval_batch_size != 0)

# Test set
test_result = test_utils.evaluate(    ##########
        tokenizer,
        tokenizer_c,
        model,
        model_c,
        test_set,
        test_set_c,
        epoch,
        batch_num,
        use_gpu,
        config,
        config_c,
        tqdm_descr='Test',
        ckpt_basename=ckpt_basename)

print(f'Accuracy: {test_result["accuracy"]}')
print('finish')
# print(f'Rouge: {test_result["rouge"]}')

with open(test_result_file, 'w') as f:
    f.write(json.dumps(test_result))
print(f'write results to {test_result_file}')
# with open(test_score_file, 'w') as f:
    # f.write(json.dumps(test_result["rouge"]))

# output for leaderboard
with open('results.out', 'w') as f:
    f.write(json.dumps(test_result["accuracy"]))
