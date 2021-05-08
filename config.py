# ==================================================================================================
# Copyright (c) 2021, Jennifer Williams and Yamagishi Laboratory, National Institute of Informatics
# Author: Jennifer Williams (j.williams@ed.ac.uk)
# All rights reserved.
# ==================================================================================================

#Please change parameters


#multi_speaker_data_path = "path/to/quantized/waveforms"
multi_speaker_data_path = "/home/s1738075/data/SIWIS/wav_norm_vqvae"
warmup_model = "pre-trained/sys2_vctk.43.upconv_702121.pyt"


sample_rate = 16000
n_fft = 1024
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 64
win_length = 1024
min_level_db = -100
ref_level_db = 20
fmin = 0
fmax = 8000
upsample_factors = (4, 4, 4)
rnn_dims=1024
fc_dims=1024
num_bit = 16
checkpoint_dir = "checkpoints/"
output_dir="output/"

batch_size=75
rnn_dims=1024
fc_dims=1024
lr=1e-3
spk_lr=0.001
num_bit = 16
num_epochs = 5000
num_workers = 0
dim_speaker_embedding = 512


# for inference only
batch_size_inference = 2
test_model_path = "checkpoints/"
test_output_dir = "inference/"
