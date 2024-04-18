#!/usr/bin/env bash

echo "Creating a new conda environment named 'anlp_hw4' with Python 3.11"
conda create -n anlp_hw4 python=3.11

echo "Activating the environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate anlp_hw4

echo "Installing PyTorch nightly preview to use MPS"
conda install pytorch-nightly::pytorch torchvision torchaudio -c pytorch-nightly

echo "Upgrading pip"
pip install --upgrade pip


echo "Installing required packages"
pip install -r requirements.txt