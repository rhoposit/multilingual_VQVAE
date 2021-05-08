sample_rate = 16000
n_fft = 1024
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 128
win_length = 1024
min_level_db = -100
ref_level_db = 20
fmin = 0
fmax = 8000
upsample_factors = (4, 4, 4)
rnn_dims=1024
fc_dims=1024
num_bit = 16
checkpoint_dir = "checkpoints"
output_dir="output"

