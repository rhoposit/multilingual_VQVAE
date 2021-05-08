#!/bin/bash
# ==================================================================================================
# Copyright (c) 2021, Jennifer Williams and Yamagishi Laboratory, National Institute of Informatics
# Author: Jennifer Williams (j.williams@ed.ac.uk)
# All rights reserved.
# ==================================================================================================
# set these parameters


#SBATCH --partition=ILCC_GPU                 # ILCC_GPU, CDT_GPU, ILCC_CPU, etc
#SBATCH --job-name=test                      # Job name
#SBATCH --ntasks=1                           # Run on a single machine
#SBATCH --gres=gpu:4                         # request N gpus
#SBATCH --cpus-per-task=1                    # require N cpus
#SBATCH --mem=14000                          # Job memory request
#SBATCH --time=2:00:00                      # Time limit hrs:min:sec
#SBATCH --output=log.out
#SBATCH --error=log.err

# Specify number of visible GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3
hostname
/usr/bin/nvidia-smi
. "/path/to/etc/profile.d/conda.sh"
conda activate project


# train a new model from scratch
# set number of gpus to match your machine
python train.py -m sys5_lang -c 1000 --gpu_devices 0 1 2 3

# load an existing model, continue training from checkpoint
# set number of gpus to match your machine
#python train.py -m sys5_lang -c 1000 -l checkpoints/sys5.43.upconv_xxxx.pyt --gpu_devices 0 1 2 3 
