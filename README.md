# multilingual_VQVAE
This is a Pytorch implementation of VQVAE mentioned in [our paper](https://arxiv.org/pdf/2105.01573).

We examine the performance of monolingual and multilingual VQVAE on four tasks: copy-synthesis, voice transformation, linguistic code-switching, and content-based masking. 

![Framework of Dual-Encoder VQVAE with one-hot global language condition](https://github.com/rhoposit/multilingual_VQVAE/blob/main/framework.png)

In brief, we have done:

1. Extended the dual-encoder VQ-VAE model with a one-hot language global condition.
2. Trained a model with multi-speaker multi-language SIWIS corpus (English, German, French, Italian).
3. Adapted a monolingual model to multilingual model.
4. Compared monolingual VCTK model with multilingual SIWIS model on 4 types of tasks.
5. We provide a SIWIS pre-trained model that uses one-hot language global condition.

# Authors 
Authors of the paper: Jennifer Williams, Jason Fong, Erica Cooper, Junichi Yamagishi

For any question related to the paper or the scripts, please contact j.williams[email mark]ed.ac.uk.

# Samples
Please find our samples [here](https://rhoposit.github.io/ssw11/index.html).

# Requirements
Please install the environment in project.yml before using the scripts.
```
conda env create -f project.yml
conda activate project
```


# Preprocessing
1. we recommend that you first trim the leading/trailing silence from the audio
2. normalize the db levels
3. use the provided pre-processing script: `preprocess_vqvae.py`
4. run `make_siwis_conditions.py` to create a proper held-out set for SIWIS data

# Usage
Please see example commands in run.sh for training the vavae model.

Or you can use python3 train.py -m [model type]. The -m option can be used to tell the the script to train a different model.

[model type] can be:
- 'sys5': train a semi-supervised VQVAE with dual encoders, and gradient reversal
- 'sys5_lang': train system 5 but use a one-hot language global condition (for multilingual)


Please modify sampling rate and other parameters in [config.py](https://github.com/rhoposit/multilingual_VQVAE/blob/main/config.py) before training.


# Pre-trained models
We offer multilingual SIWIS pre-trained models for system 5 (sys5_lang)


# Multi-gpu parallel training
Please see example commands in run_slurm.sh for running on SLURM with multiple GPUs


# Acknowledgement

The code is based on [mkotha/WaveRNN](https://github.com/mkotha/WaveRNN)

And is also based on [nii-yamagishilab/Extended_VQVAE](https://github.com/nii-yamagishilab/Extended_VQVAE)

And is also based on [rhoposit/icassp2021](https://github.com/rhoposit/icassp2021)


This work was partially supported by the EPSRC Centre for DoctoralTraining in Data Science, funded by the UK Engineering and Physical Sci-ences Research Council (grant EP/L016427/1) and University of Edinburgh;and by a JST CREST Grant (JPMJCR18A6, VoicePersonae project), Japan.The numerical calculations were carried out on the TSUBAME 3.0 super-computer at the Tokyo Institute of Technology.

# License

MIT License
- Copyright (c) 2019, fatchord (https://github.com/fatchord)
- Copyright (c) 2019, mkotha (https://github.com/mkotha)
- Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics.
- Copyright (c) 2021, Jennifer Williams and Yamagishi Laboratory, National Institute of Informatics.



Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.