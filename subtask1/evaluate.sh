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


# Evaluate trained model on test set and print results to "results.out"
# python test.py --test test.txt --config configs/10.10/xlmr_aug_v3.2_attack_pgd_remove_times2_len250.json --model trained_model/baseline_aug_v3.2/xlmr_aug_v3.2_attack_pgd_remove_times2_len250/version_0 --use_tongxiao_post --use_post --post_method 125 --online_submission_test
python integrator.py --config configs/xlmr_aug_v3.2_integarator.json --use_tongxiao_post --use_post --post_method 125 --online_submission_test
