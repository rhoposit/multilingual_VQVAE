# ==================================================================================================
# Copyright (c) 2021, Jennifer Williams and Yamagishi Laboratory, National Institute of Informatics
# Author: Jennifer Williams (j.williams@ed.ac.uk)
# All rights reserved.
# ==================================================================================================
# set these parameters


# Specify number of visible GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3
hostname
/usr/bin/nvidia-smi
. "/path/to/etc/profile.d/conda.sh"
conda activate project


# train a new model from scratch
# set number of gpus to match your machine
python train.py -m sys5 -c 1000 --gpu_devices 0 1 2 3

# load an existing model, continue training from checkpoint
# set number of gpus to match your machine
python train.py -m sys5 -c 1000 -l checkpoints/sys5.43.upconv_xxxx.pyt --gpu_devices 0 1 2 3 
