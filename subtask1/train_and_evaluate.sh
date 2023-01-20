#!/bin/bash

# Using conda within a shell script
source ~/miniconda3/etc/profile.d/conda.sh

yes | conda create --name nl4opt-Long python=3.9.12
conda activate nl4opt-Long


# Ensure the correct interpreter is executed
echo $(which python)
echo $(which pip)


# Install all other dependencies
pip install --no-cache-dir --ignore-installed -r requirements.txt
pip install en_core_web_sm-3.2.0.tar.gz

# Upgrade Pytorch for CUDA 11.6
pip install --upgrade --force-reinstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# training all four models
# run model 1
python train.py --config configs/10.9/xlmr_aug_v3.2_attack_fgm_remove_times.json --online_submission_train --train data/train/train_aug_v3.2.txt --dev data/dev/dev.txt --test test.txt --decreasing_lr --seed 42
# run model 2
python train.py --config configs/10.9/xlmr_aug_v3.2_attack_pgd_stand_remove_times2.json --online_submission_train --train data/train/train_aug_v3.2.txt --dev data/dev/dev.txt --test test.txt --decreasing_lr --seed 42
# run model 3
python train.py --config configs/10.10/xlmr_aug_v3.2_attack_pgd_remove_times2_len250.json --online_submission_train --train data/train/train_aug_v3.2.txt --dev data/dev/dev.txt --test test.txt --seed 42
# run model 4
python train.py --config configs/10.11/xlmr_large_aug_v3.2_attack_pgd_stand_remove_times.json --online_submission_train --train data/train/train_aug_v3.2.txt --dev data/dev/dev.txt --test test.txt --seed 42
python fine.py --config configs/10.11/xlmr_large_aug_v3.2_attack_pgd_stand_remove_times.json  --model trained_model/baseline_aug_v3.2_new/xlmr_large_aug_v3.2_attack_pgd_stand_remove_times/version_0 --online_submission_train --lr 5e-6 --train data/train/train_aug_v3.2.txt --dev data/dev/dev.txt --test test.txt --seed 42

# Evaluate trained model on test set and print results to "results.out"

# the final ensembled model for prediction
# note:  the training pipline is a little long (about 15hours), we can't comfirm it would breakdown in new enviroment, so
python integrator.py --config configs/xlmr_aug_v3.2_integarator.json --use_tongxiao_post --use_post --post_method 125 --online_submission_test --online_submission_train  --seed 45
