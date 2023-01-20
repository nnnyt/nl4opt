import os
import time
from tqdm import tqdm
import numpy as np
from collections import OrderedDict, defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .model_cascade import CascadeNerModel
from .model_mrc import MRCNerModel
from .train_utils import save_wise_model, load_wise_model, get_pretrained_model_path, set_model_dir
from .attack_utils import FGM, PGD

class CustomTrainer(object):
    def __init__(self, model, 
                train_dataloader=None, valid_dataloader=None, test_dataloader=None,
                
                out_model_dir=None, 
                out_model_name=None,

                epochs=None,
                grad_accum=None,

                use_attack_method=None,
                adversarial_params={},

                clip_grad_norm=None,
        ):
        self.model = model
        self.train_dataloader=train_dataloader
        self.valid_dataloader=valid_dataloader
        self.test_dataloader=test_dataloader

        self.out_model_dir = out_model_dir
        self.out_model_name = out_model_name

        self.epochs = epochs
        self.grad_accum = grad_accum

        if self.out_model_dir and self.out_model_name:
            self.bcp_path = set_model_dir(self.out_model_dir)
            self.writer = SummaryWriter(os.path.join(self.bcp_path, "log/"))
        self._global_step = 0

        self.clip_grad_norm = clip_grad_norm

        self.use_attack_method = use_attack_method
        self.adversarial = {
            'name': use_attack_method,
            "emb_name": "word_embeddings",
        }
        self.adversarial.update(adversarial_params)

    def adversarial_initialize(self):
        '''对抗训练初始化
        '''
        assert self.adversarial['name'] in {'', 'fgm', 'pgd', 'vat', 'gradient_penalty'}, 'adversarial_train support fgm, pgd, vat and gradient_penalty mode'
        self.adversarial['epsilon'] = self.adversarial.get('epsilon', 1.0)
        self.adversarial['emb_name'] = self.adversarial.get('emb_name', 'word_embeddings')

        if self.adversarial['name'] == 'fgm':
            self.ad_train = FGM(self.model)
        elif self.adversarial['name'] == 'pgd':
            self.adversarial['K'] = self.adversarial.get('K', 3)  # 步数
            self.adversarial['alpha'] = self.adversarial.get('alpha', 0.3)  # 学习率
            self.ad_train = PGD(self.model)

    def adversarial_training(self, model, batch, use_post=True):
        '''对抗训练
        '''
        if self.adversarial['name'] == 'fgm':
            self.ad_train.attack(**self.adversarial) # embedding被修改了

            attack_output = model(batch, mode="train", use_post=use_post)
            
            attack_loss = attack_output["loss"] / self.grad_accum
            attack_loss.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度
            # 恢复Embedding的参数, 因为要在正常的embedding上更新参数，而不是增加了对抗扰动后的embedding上更新参数~
            self.ad_train.restore(**self.adversarial)
        
        elif self.adversarial['name'] == 'pgd':
            self.ad_train.backup_grad()  # 备份梯度
            for t in range(self.adversarial['K']):
                # 在embedding上添加对抗扰动, first attack时备份param.data
                self.ad_train.attack(**self.adversarial, is_first_attack=(t==0))
                if t != self.adversarial['K']-1:
                    self.optimizer.zero_grad()  # 为了累积扰动而不是梯度
                else:
                    self.ad_train.restore_grad() # 恢复正常的grad
                
                attack_output = model(batch, mode="train", use_post=use_post)

                attack_loss = attack_output["loss"] / self.grad_accum
                attack_loss.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度

            self.ad_train.restore(**self.adversarial) # 恢复embedding参数

    def setup_env(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def log(self, key, value):
        self.writer.add_scalar(key, value, global_step=self._global_step)

    def write_metrics(self, pred_results, loss=0.0, suffix=''):
        self.log(suffix + 'loss', loss)
        for key in pred_results:            
            # 将 F1 、P 、 R 至于同一分组
            key_group = key.split("@")[-1]
            self.log(f"{suffix}{key_group}/{suffix}{key}", pred_results[key])

    def train(self, model=None, train_dataloader=None, use_post=False, post_method=[]):
        if use_post:
            print("[DEBUG] Trainer.train use_post=", use_post)
        self._global_step = 0
        if self.use_attack_method:
            self.adversarial_initialize()

        model = self.model if model is None else model
        train_dataloader = self.train_dataloader if train_dataloader is None else train_dataloader

        model.train()
        best_metric = -1
        for epoch in range(self.epochs):
            losses = []
            num_step_per_epoch = len(train_dataloader)
            process_bar = tqdm(enumerate(train_dataloader), total=num_step_per_epoch, mininterval=1)
            for idx, batch in process_bar:
                model.train() # 设置为训练模式（test 和 valid 会影响训练）

                # tokens, tags, mask, token_mask, metadata, pos_tag_idx, sentence_str = batch
                tr_output = model(batch, mode="train", use_post=use_post, post_method=post_method)

                # 分析各个标签
                for key in ["F1@CONST_DIR", "F1@LIMIT", "F1@OBJ_DIR", "F1@OBJ_NAME", "F1@PARAM", "F1@VAR", "micro@F1"]:
                    if key in tr_output['results']:
                        self.writer.add_scalars("train_F1_spans", {key: tr_output['results'][key]}, self._global_step)
                
                if isinstance(model, CascadeNerModel):
                    # if epoch < 10:
                    #     p_border = 0.9
                    # elif epoch < 20:
                    #     p_border = 0.1
                    # else:
                    #     p_border = 0.5
                    p_border = 0.5
                    r_border = 1
                    r_tag = 1
                    loss = p_border * tr_output["border_loss"] * r_border + (1 - p_border) * tr_output["tag_loss"] * r_tag
                elif isinstance(model, MRCNerModel):
                    p_start = 0.5
                    r_start = 1
                    r_end = 1
                    loss = p_start * tr_output["start_loss"] * r_start + r_end * (1 - p_start) * tr_output["end_loss"]
                else:
                    loss = tr_output['loss']
                
                # Log Epoch
                self.write_metrics(tr_output['results'], loss=loss.item(), suffix='')
                losses.append(loss.item())
                
                # 反向传播梯度
                loss = loss / self.grad_accum        # 损失标准化
                loss.backward()

                # 对抗训练
                self.adversarial_training(model, batch, use_post=use_post)

                # 梯度裁剪
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
                
                # 梯度积累 -> 优化
                if (idx+1) % self.grad_accum == 0:
                    self.optimizer.step()                    # 更新参数
                    self.optimizer.zero_grad()
                
                # change lr scheduler
                if self.scheduler is not None:
                    # warmingup
                    if not isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step()
                
                self._global_step += 1
                if isinstance(model, CascadeNerModel):
                    process_bar.set_postfix(OrderedDict(loss=loss.item(), border_loss=tr_output["border_loss"].item(), tag_loss=tr_output["tag_loss"].item(), **tr_output['results']))
                    self.writer.add_scalars("epoch/tag_loss", {"Train": tr_output["tag_loss"]}, epoch)
                    self.writer.add_scalars("epoch/border_loss", {"Train": tr_output["border_loss"]}, epoch)
                elif isinstance(model, MRCNerModel):
                    process_bar.set_postfix(OrderedDict(loss=loss.item(), end_loss=tr_output["end_loss"].item(), start_loss=tr_output["start_loss"].item(), **tr_output['results']))
                    self.writer.add_scalars("epoch/start_loss", {"Train": tr_output["start_loss"]}, epoch)
                    self.writer.add_scalars("epoch/end_loss", {"Train": tr_output["end_loss"]}, epoch)
                else:
                    process_bar.set_postfix(OrderedDict(loss=loss.item(), **tr_output['results']))
                process_bar.set_description(f"Epoch {epoch}: ")

                
            print(f"=========== Finish Training Epoch {epoch} ================")
            # Log Epoch
            pred_results = model.span_f1.get_metric(True)
            tr_loss = np.mean(losses)
            self.writer.add_scalars("epoch/loss", {"Train": tr_loss}, epoch)
            self.writer.add_scalars("epoch/span_f1", {"Train": pred_results["micro@F1"]}, epoch)

            if isinstance(model, CascadeNerModel):
                pred_border_results = model.border_f1.get_metric(True)
                self.writer.add_scalars("epoch/border_f1", {"Train": pred_border_results["micro@F1"]}, epoch)

            # 评估
            # val_out = self.valid(use_post=True) # 按后处理的结果保存最好的模型
            val_out = self.valid(use_post=use_post) # 和 训练保持一致

            val_metric = val_out["results"]["micro@F1"]
            val_loss = val_out["val_loss"]
            self.writer.add_scalars("epoch/loss", {"Valid": val_loss}, epoch)
            self.writer.add_scalars("epoch/span_f1", {"Valid": val_metric}, epoch)
            
            if isinstance(model, CascadeNerModel):
                self.writer.add_scalars("epoch/border_f1", {"Valid": val_out["border_results"]["micro@F1"]}, epoch)
                self.writer.add_scalars("epoch/tag_loss", {"Valid": val_out["tag_loss"]}, epoch)
                self.writer.add_scalars("epoch/border_loss", {"Valid": val_out["border_loss"]}, epoch)
            elif isinstance(model, MRCNerModel):
                self.writer.add_scalars("epoch/start_loss", {"Valid": val_out["start_loss"]}, epoch)
                self.writer.add_scalars("epoch/end_loss", {"Valid": val_out["end_loss"]}, epoch)

            # 分析各个标签
            for key in ["F1@CONST_DIR", "F1@LIMIT", "F1@OBJ_DIR", "F1@OBJ_NAME", "F1@PARAM", "F1@VAR", "micro@F1"]:
                if key in val_out['results']:
                    self.writer.add_scalars("epoch/val_F1_spans", {key: val_out['results'][key]}, self._global_step)
            
            if epoch % 5 == 0:
                self.test(use_post=use_post, verbose=False) # 和 训练 验证 保持一致
                print("<-- test with use_post=", use_post)
                # self.test(use_post=True, verbose=False) # 用于观察
                # print("<-- test with use_post=True")
            
            # sava best model
            if val_metric > best_metric:
                best_metric = val_metric
                
                old_model_path = get_pretrained_model_path(self.bcp_path)
                if old_model_path and os.path.exists(old_model_path):
                    os.remove(old_model_path)

                timestamp = time.time()
                filename = 'model_' + str(timestamp) + f'_epoch{epoch}_val_micro@F1{val_metric:.4f}_best.bin'
                model.save_pretrained(self.bcp_path, filename)

            # change lr scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
        self.writer.close()
        print("=========== Finish Training ================")

    def valid(self, model=None, valid_dataloader=None, use_post=False, post_method=[]):
        if use_post:
            print("[DEBUG] Trainer.valid use_post=", use_post)
        model = self.model if model is None else model
        valid_dataloader = self.valid_dataloader if valid_dataloader is None else valid_dataloader

        model.eval()
        losses = []
        for idx, batch in enumerate(valid_dataloader):
            # tokens, tags, mask, token_mask, metadata, pos_tag_idx, sentence_str = batch
            output = model(batch, mode="dev", use_post=use_post)
            if isinstance(model, CascadeNerModel):
                loss = output["loss"] if "loss" in output else (output["border_loss"] + output["tag_loss"])
            elif isinstance(model, MRCNerModel):
                loss = output["loss"] if "loss" in output else (output["start_loss"] + output["end_loss"])
            else:
                loss = output["loss"]
            losses.append(loss.item())
        
        # Log Epoch
        pred_results = model.span_f1.get_metric(True)
        if isinstance(model, CascadeNerModel):
            pred_border_results = model.border_f1.get_metric(True)
        
        avg_loss = np.mean(losses)

        print("="*100)
        print("="*100)
        for key in ["F1@CONST_DIR", "F1@LIMIT", "F1@OBJ_DIR", "F1@OBJ_NAME", "F1@PARAM", "F1@VAR", "micro@F1"]:
            if key in pred_results:
                _key =  f"val_{key}"
                print(f"{_key}:   {pred_results[key]}")
        print("="*100)
        print("="*100)

        
        if isinstance(model, CascadeNerModel):
            out = {"val_loss": avg_loss, "results": pred_results, "tag_loss": output["tag_loss"], "border_loss": output["border_loss"]}
            out["border_results"] = pred_border_results
        elif isinstance(model, MRCNerModel):
            out = {"val_loss": avg_loss, "results": pred_results, "start_loss": output["start_loss"], "end_loss": output["end_loss"]}
        else:
            out = {"val_loss": avg_loss, "results": pred_results}
        return out

    def test(self, model=None, test_dataloader=None, use_post=True, post_method=[1, 2], verbose=True):
        if use_post:
            print("[DEBUG] Trainer.test use_post=", use_post)
        model = self.model if model is None else model
        test_dataloader = self.test_dataloader if test_dataloader is None else test_dataloader
        
        model.eval()

        losses = []

        type_to_ignore_dict = {
            key: defaultdict(int) for key in ["CONST_DIR", "LIMIT", "OBJ_DIR", "OBJ_NAME", "PARAM", "VAR"]
        }
        type_to_wrong_dict = {
            key: defaultdict(int) for key in ["CONST_DIR", "LIMIT", "OBJ_DIR", "OBJ_NAME", "PARAM", "VAR"]
        }

        for idx, batch in enumerate(test_dataloader):
            # tokens, tags, mask, token_mask, metadata, pos_tag_idx, sentence_str = batch
            if idx == 285:
                print("DEBUG")
            output = model(batch, mode="predict", use_post=use_post, post_method=post_method, verbose=verbose)
            # Log Epoch
            loss = output["loss"] if "loss" in output else (output["border_loss"] + output["tag_loss"])
            losses.append(loss.item())
            
            if verbose:
                """Important !!! Make batch_size = 1 if log debug info"""
                print(f"############# [batch_idx {idx}] #############")
                print("sentence_str : ", batch[-1])

                tok_list, gold_tok_tag_list = batch[-1][0].split(), batch[1][0].cpu().detach().numpy().tolist()[1:-1]
                print("gold_token_tags : ", [(i, tok, tag) for i, (tok, tag) in enumerate(zip(tok_list, [model.id_to_tag[x] for x in gold_tok_tag_list]))] )
                print()

                gold_spans = [(" ".join(tok_list[k[0]-1: k[1]+1-1]), k, v) for k, v in output["predictions"]["gold_spans"][0].items()]
                pred_spans = [(" ".join(tok_list[k[0]-1: k[1]+1-1]), k, v) for k, v in output["predictions"]["pred_spans"][0].items()]
                print("gold_spans : ", gold_spans)
                print("pred_spans : ", pred_spans)
                print()
                gold_spans_set = set(gold_spans)
                pred_spans_set = set(pred_spans)
                diff_1  = list(pred_spans_set - gold_spans_set)
                diff_1 = sorted(diff_1,key=lambda x: x[1][0])
                print(f"pred - gold ( {len(diff_1)} ) : {diff_1}")
                for item in diff_1:
                    if item[2] == "O":
                        continue
                    type_to_wrong_dict[ item[2] ][ item[0] ] += 1
                print()
                diff_2 = list(gold_spans_set - pred_spans_set)
                diff_2 = sorted(diff_2,key=lambda x: x[1][0])
                print(f"gold - pred ( {len(diff_2)} ) : {diff_2}")
                for item in diff_2:
                    if item[2] == "O":
                        continue
                    type_to_ignore_dict[ item[2] ][ item[0] ] += 1
                print("="*100)
        # Log Epoch
        pred_results = model.span_f1.get_metric(True)
        avg_loss = np.mean(losses)

        if verbose:
            print("type_to_wrong_dict")
            print(type_to_wrong_dict)
            print()
            print("type_to_ignore_dict")
            print(type_to_ignore_dict)
            print()

        out = {"test_loss": avg_loss, "results": pred_results}
        print(out)
        print("="*100)
        print("="*100)
        for key in ["F1@CONST_DIR", "F1@LIMIT", "F1@OBJ_DIR", "F1@OBJ_NAME", "F1@PARAM", "F1@VAR", "micro@F1"]:
            if key in pred_results:
                _key =  f"test_{key}"
                print(f"{_key}:   {pred_results[key]}")
        print("="*100)
        print("="*100)
        return out

    def predict_tags(self, model=None, test_dataloader=None, use_post=False):
        if use_post:
            print("[DEBUG] Trainer.predict_tags use_post=", use_post)
        model = self.model if model is None else model
        test_dataloader = self.test_dataloader if test_dataloader is None else test_dataloader

        model.eval()

        out_str = ''
        index = 0
        for idx, batch in tqdm(enumerate(test_dataloader)):
            pred_tags = model.predict_tags(batch, use_post=use_post)

            for pred_tag_inst in pred_tags:
                out_str += '\n'.join(pred_tag_inst)
                out_str += '\n\n'
            index += 1
        return out_str

