#!/bin/bash

# Using conda within a shell script
source ~/miniconda3/etc/profile.d/conda.sh

# # Create and activate conda environment
conda env create -f environment.yml
conda activate tgen

# Ensure the correct interpreter is executed
echo $(which python)
echo $(which pip)


# Upgrade Pytorch for CUDA 11.6
pip install --upgrade --force-reinstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116


# Train
python train_attack.py --config configs/prompt-in-attack_fix_const_2.json

python train_span.py --config configs/span-in.json