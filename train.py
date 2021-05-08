# ==================================================================================================
# Copyright (c) 2021, Jennifer Williams and Yamagishi Laboratory, National Institute of Informatics
# Author: Jennifer Williams (j.williams@ed.ac.uk)
# All rights reserved.
# ==================================================================================================


import math, os, sys
import pickle5 as pickle
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import *
import sys
import utils.env as env
import argparse
import platform
import re
import utils.logger as logger
import time
import subprocess
import pytorch_warmup as warmup
from collections import defaultdict
import models.sys5 as sys5_model
import models.sys5_lang as sys5_model_lang
import config as config

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import operator
from functools import reduce

parser = argparse.ArgumentParser(description='Train or run some neural net')
parser.add_argument('--generate', '-g', action='store_true')
parser.add_argument('--float', action='store_true')
parser.add_argument('--half', action='store_true')
parser.add_argument('--load', '-l')
parser.add_argument('--scratch', action='store_true')
parser.add_argument('--model', '-m')
parser.add_argument('--force', action='store_true', help='skip the version check')
parser.add_argument('--count', '-c', type=int, default=1, help='size of the test set')
parser.add_argument('--partial', action='append', default=[], help='model to partially load')

parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
args = parser.parse_args()


gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

def get_dataset(dataset_type, config, model_type):
    if dataset_type == 'vctk':
        data_path = config.multi_speaker_data_path
        with open(f'{data_path}/index.pkl', 'rb') as f:
            index = pickle.load(f)
        train_set = index[:-10]
        unseen_set = index[-10:]
        test_index = [x[:10] if i < 2 * args.count else [] for i, x in enumerate(train_set)]
        train_index = [x[10:-1] if i < args.count else x for i, x in enumerate(train_set)]
        unseen_index = [x for i, x in enumerate(unseen_set)]
        print("Train set speakers: ", len(train_index))
        print("Test set speakers: ", len(test_index))
        print("Held out test set speakers: ", len(unseen_index))
        dataset = env.MultispeakerDataset(train_index, data_path)
        spk_path = None

    # use the index2.pkl that filters out unseen speakers
    elif dataset_type == 'siwis_lang':
        data_path = config.multi_speaker_data_path
        with open(f'{data_path}/index2.pkl', 'rb') as f:
            index = pickle.load(f)
        train_set = index[:]
        test_index = [x[:75] for i, x in enumerate(train_set)]
        train_index = [x[75:] for i, x in enumerate(train_set)]
        print("Train set speakers: ", len(train_index))
        print("Test set speakers: ", len(test_index))
        print("Test set total: ", sum( [ len(listElem) for listElem in test_index]))
        dataset = env.MultispeakerDataset_lang(train_index, data_path)
        spk_path = None
    else:
        raise RuntimeError(dataset_type, 'bad dataset type')
    return data_path, spk_path, index, test_index, train_index, dataset


print('Get Parameters')
def main():
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node:{}".format(ngpus_per_node))

    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    print('==> Making model..')

    if args.float and args.half:
        sys.exit('--float and --half cannot be specified together')

    if args.float:
        use_half = False
    elif args.half:
        use_half = True
    else:
        use_half = False

    model_type = args.model
    model_name = f'{model_type}.43.upconv'

    vctk_num_spk = 109

    
    if model_type == 'sys5':
        dataset_type = 'vctk'
        data_path, spk_path, index, test_index, train_index, dataset = get_dataset(dataset_type,config, model_type)
        # load a warmed up model
        warm = sys5_model.Model(nspeakers=dataset.num_speakers(),rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, global_decoder_cond_dims=dataset.num_speakers(),upsample_factors=config.upsample_factors, normalize_vq=True, noise_x=True, noise_y=True)
        model_fn = env.restore_warmup(warm, config.warmup_model)

    elif model_type == 'sys5_lang':
        dataset_type = 'siwis_lang'
        lang_dims = 4
        nspeakers = 31
        data_path, spk_path, index, test_index, train_index, dataset = get_dataset(dataset_type,config, model_type)
        # train from scratch, use a language code
        warm = sys5_model_lang.Model(lang_dims, nspeakers=nspeakers,rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, global_decoder_cond_dims=nspeakers,upsample_factors=config.upsample_factors, normalize_vq=True, noise_x=True, noise_y=True)
        model_fn = env.restore_warmup(warm, config.warmup_model)
        
    else:
        sys.exit(f'Unknown model: {model_type}')


    print(f'dataset size: {len(dataset)}')
    ###############update line##################
    model = model_fn
    paths = env.Paths(model_name, data_path, spk_path)
    
    if args.scratch or args.load == None and not os.path.exists(paths.model_path()):
        # Start from scratch
        step = 0
    else:
        if args.load:
            prev_model_name = re.sub(r'_[0-9]+$', '', re.sub(r'\.pyt$', '', os.path.basename(args.load)))
            prev_model_basename = prev_model_name.split('_')[0]
            model_basename = model_name.split('_')[0]
            if prev_model_basename != model_basename and not args.force:
                sys.exit(f'refusing to load {args.load} because its basename ({prev_model_basename}) is not {model_basename}')
            if args.generate:
                paths = env.Paths(prev_model_name, data_path)
            prev_path = args.load
        else:
            prev_path = paths.model_path()
        step = env.restore(prev_path, model, args.gpu)

    optimiser = optim.AdamW(model.parameters(), betas=(0.9, 0.999), weight_decay=0.01)


    if args.generate:
        model.do_generate(paths, step, data_path, test_index, use_half=use_half, verbose=True)#, deterministic=True)
    else:
        logger.set_logfile(paths.logfile_path())
        logger.log('------------------------------------------------------------')
        logger.log('-- New training session starts here ------------------------')
        logger.log(time.strftime('%c UTC', time.gmtime()))
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        config.batch_size = int(config.batch_size / ngpus_per_node)
        config.num_workers = int(config.num_workers / ngpus_per_node)
        print("BATCH SIZE: ", config.batch_size)
        print("NUM WORKERS: ", config.batch_size)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('The number of parameters of model is', num_params)

        
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        model.module.do_train(paths, dataset, optimiser, epochs=config.num_epochs, batch_size=config.batch_size, num_workers=config.num_workers, step=step, train_sampler=train_sampler, device=args.gpu,lr=config.lr, spk_lr=config.spk_lr, use_half=use_half, valid_index=test_index)

if __name__ == '__main__':
    main()
        
