from torch.utils.data import Dataset
import torch
import os
import numpy as np
from utils.dsp import *
import re
import config
import pandas as pd
from torchvision import transforms
import h5py
from torch._six import container_abcs, string_classes, int_classes
import itertools
import random 

bits = 16
seq_len = config.hop_length * 5

class Paths:
    def __init__(self, name, data_dir, spk_dir, checkpoint_dir=config.checkpoint_dir, output_dir=config.output_dir):
        self.name = name
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.spk_dir = spk_dir

    def model_path(self):
        return f'{self.checkpoint_dir}/{self.name}.pyt'

    def model_hist_path(self, step):
        return f'{self.checkpoint_dir}/{self.name}_{step}.pyt'

    def step_path(self):
        return f'{self.checkpoint_dir}/{self.name}_step.npy'

    def gen_path(self):
        return f'{self.output_dir}/{self.name}/'

    def logfile_path(self):
        return f'log/{self.name}'

def default_paths(name, data_dir):
    return Paths(name, data_dir, spk_dir, checkpoint_dir=config.checkpoint_dir, output_dir=config.output_dir)

class AudiobookDataset(Dataset):
    def __init__(self, ids, path):
        self.path = path
        self.metadata = ids

    def __getitem__(self, index):
        file = self.metadata[index]
        m = np.load(f'{self.path}/mel/{file}.npy')
        x = np.load(f'{self.path}/quant/{file}.npy')
        return m, x

    def __len__(self):
        return len(self.metadata)

class MultispeakerDataset(Dataset):
    def __init__(self, index, path):
        self.path = path
        self.index = index
        self.all_files = [(i, name) for (i, speaker) in enumerate(index) for name in speaker]
        np.random.shuffle(self.all_files)

    def __getitem__(self, index):
        speaker_id, name = self.all_files[index]
        speaker_onehot = (np.arange(len(self.index)) == speaker_id).astype(np.long)
        path = f'{self.path}/{name}.npy'
        audio = np.load(f'{self.path}/{name}.npy')
        return speaker_onehot, audio, path

    def __len__(self):
        return len(self.all_files)

    def num_speakers(self):
        return len(self.index)



class MultispeakerDataset_lang(Dataset):
    def __init__(self, index, path):
        self.path = path
        self.index = index
        self.langs = {"EN":0, "FR":1, "IT":2, "DE":3}
        self.all_files = [(i, name) for (i, speaker) in enumerate(index) for name in speaker]
        np.random.shuffle(self.all_files)

    def __getitem__(self, index):
        speaker_id, name = self.all_files[index]
        language_id = self.langs[name.split("_")[0]]
        language_onehot = (np.arange(len(self.langs)) == language_id).astype(np.long)
        speaker_onehot = (np.arange(len(self.index)) == speaker_id).astype(np.long)
        path = f'{self.path}/{name}.npy'
        audio = np.load(f'{self.path}/{name}.npy')
        return speaker_onehot, language_onehot, audio, path

    def __len__(self):
        return len(self.all_files)

    def num_speakers(self):
        return len(self.index)


    

    
class MultiWaveRNNDataset(Dataset):
    def __init__(self, index, path, num_spk):
        self.path = path
        self.index = index
        self.num_spk = num_spk
        self.all_files = [(i, name) for (i, speaker) in enumerate(index) for name in speaker]

    def __getitem__(self, index):
        speaker_id, name = self.all_files[index]
        speaker_onehot = (np.arange(self.num_spk) == speaker_id).astype(np.long)
        mel = np.load(f'{self.path}/mel/{name}.npy')
        audio = np.load(f'{self.path}/quant/{name}.npy')

        # T = mel.shape[-1]
        # speaker = np.tile(speaker_onehot, (T, 1)).T
        # print(speaker.shape, mel.shape, audio.shape)
        # exit ()
        return speaker_onehot, mel, audio, name

    def __len__(self):
        return len(self.all_files)

    def num_speakers(self):
        return len(self.index)

    
    


def collate_multiWaveRNN_samples(left_pad, mel_win, right_pad, batch):
    speakers_onehot = [x[0] for x in batch]
    max_offsets = [x[1].shape[-1] - mel_win for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [offset * hop_length for offset in mel_offsets]
    mels = [x[1][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]
    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x[2], np.zeros(right_pad, dtype=np.int16)])[sig_offsets[i]:sig_offsets[i] + left_pad + 64 * mel_win + right_pad] for i, x in enumerate(batch)]
    mels = np.stack(mels).astype(np.float32)
    wave16 = np.stack(wave16).astype(np.int64) + 2**15
    coarse = wave16 // 256
    fine = wave16 % 256
    speakers_onehot = torch.FloatTensor(speakers_onehot)
    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)
    fine = torch.LongTensor(fine)
    coarse_f = coarse.float() / 127.5 - 1.
    fine_f = fine.float() / 127.5 - 1.
    return speakers_onehot, mels, coarse, fine, coarse_f, fine_f


def collate_multiWaveRNN_samples_forward(left_pad, mel_win, right_pad, batch):
    speakers_onehot = [x[0] for x in batch]
    names = [x[-1] for x in batch]
    max_offsets = [x[1].shape[-1] - mel_win for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [offset * hop_length for offset in mel_offsets]
    mels = [x[1][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]
    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x[2], np.zeros(right_pad, dtype=np.int16)])[sig_offsets[i]:sig_offsets[i] + left_pad + 64 * mel_win + right_pad] for i, x in enumerate(batch)]
    mels = np.stack(mels).astype(np.float32)
    wave16 = np.stack(wave16).astype(np.int64) + 2**15
    coarse = wave16 // 256
    fine = wave16 % 256
    speakers_onehot = torch.FloatTensor(speakers_onehot)
    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)
    fine = torch.LongTensor(fine)
    coarse_f = coarse.float() / 127.5 - 1.
    fine_f = fine.float() / 127.5 - 1.
    return speakers_onehot, mels, coarse, fine, coarse_f, fine_f, names
    

def collate_multispeaker_samples(left_pad, window, right_pad, batch):
    samples = [x[1] for x in batch]
    speakers_onehot = torch.FloatTensor([x[0] for x in batch])
    max_offsets = [x.shape[-1] - window for x in samples]
    offsets = [np.random.randint(0, offset) for offset in max_offsets]

    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x, np.zeros(right_pad, dtype=np.int16)])[offsets[i]:offsets[i] + left_pad + window + right_pad] for i, x in enumerate(samples)]
    return torch.FloatTensor(speakers_onehot), torch.LongTensor(np.stack(wave16).astype(np.int64))



def collate_multispeaker_samples_forward(left_pad, window, right_pad, batch):
    samples = [x[1] for x in batch]
    paths = [x[2] for x in batch]
    speakers_onehot = torch.FloatTensor([x[0] for x in batch])
    max_offsets = [x.shape[-1] - window for x in samples]
    offsets = [np.random.randint(0, offset) for offset in max_offsets]

    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x, np.zeros(right_pad, dtype=np.int16)])[offsets[i]:offsets[i] + left_pad + window + right_pad] for i, x in enumerate(samples)]
    return torch.FloatTensor(speakers_onehot), torch.LongTensor(np.stack(wave16).astype(np.int64)), paths



def collate_multispeaker_samples_lang(left_pad, window, right_pad, batch):
    samples = [x[2] for x in batch]
    speakers_onehot = torch.FloatTensor([x[0] for x in batch])
    lang_onehot = torch.FloatTensor([x[1] for x in batch])
    max_offsets = [x.shape[-1] - window for x in samples]
    offsets = [np.random.randint(0, offset) for offset in max_offsets]

    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x, np.zeros(right_pad, dtype=np.int16)])[offsets[i]:offsets[i] + left_pad + window + right_pad] for i, x in enumerate(samples)]
    return torch.FloatTensor(speakers_onehot), torch.FloatTensor(lang_onehot), torch.LongTensor(np.stack(wave16).astype(np.int64))



def collate_multispeaker_samples_lang_forward(left_pad, window, right_pad, batch):
    samples = [x[2] for x in batch]
    paths = [x[3] for x in batch]
    speakers_onehot = torch.FloatTensor([x[0] for x in batch])
    lang_onehot = torch.FloatTensor([x[1] for x in batch])
    max_offsets = [x.shape[-1] - window for x in samples]
    offsets = [np.random.randint(0, offset) for offset in max_offsets]

    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x, np.zeros(right_pad, dtype=np.int16)])[offsets[i]:offsets[i] + left_pad + window + right_pad] for i, x in enumerate(samples)]
    return torch.FloatTensor(speakers_onehot), torch.FloatTensor(lang_onehot), torch.LongTensor(np.stack(wave16).astype(np.int64)), paths



def collate_samples(left_pad, window, right_pad, batch):
    #print(f'collate: window={window}')
    samples = [x[1] for x in batch]
    max_offsets = [x.shape[-1] - window for x in samples]
    offsets = [np.random.randint(0, offset) for offset in max_offsets]

    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x, np.zeros(right_pad, dtype=np.int16)])[offsets[i]:offsets[i] + left_pad + window + right_pad] for i, x in enumerate(samples)]
    return torch.LongTensor(np.stack(wave16).astype(np.int64))

def collate(left_pad, mel_win, right_pad, batch) :
    max_offsets = [x[0].shape[-1] - mel_win for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [offset * hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x[1], np.zeros(right_pad, dtype=np.int16)])[sig_offsets[i]:sig_offsets[i] + left_pad + hop_length * mel_win + right_pad] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    wave16 = np.stack(wave16).astype(np.int64) + 2**15
    coarse = wave16 // 256
    fine = wave16 % 256

    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)
    fine = torch.LongTensor(fine)

    coarse_f = coarse.float() / 127.5 - 1.
    fine_f = fine.float() / 127.5 - 1.

    return mels, coarse, fine, coarse_f, fine_f






def pad_to_length(x, m):
    return np.pad(x,((0, 0), (0, m - x.shape[1])), mode = 'constant')



def collate_mfcc(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    data = [np.array(b[0], dtype=np.float32) for b in batch]
    speakers_onehot = torch.FloatTensor([x[1] for x in batch])
    max_val = max([x.shape[1] for x in data])
    padded = [pad_to_length(x, max_val) for x in data]
    mfccs = torch.FloatTensor(np.stack(padded).astype(np.int64))
    return mfccs, torch.FloatTensor(speakers_onehot)



def restore(path, model, gpu):
    loc = 'cuda:{}'.format(gpu)
    # checkpoint = torch.load(prev_path)
    # model.load_state_dict(checkpoint['state_dict'])
    # print('loaded model successfully!')

    model.load_state_dict(torch.load(path, map_location=loc))
    print('reload model successfully!')

    match = re.search(r'_([0-9]+)\.pyt', path)
    if match:
        return int(match.group(1))

    step_path = re.sub(r'\.pyt', '_step.npy', path)
    return np.load(step_path)




def restore_warmup(model, model_path):
    #state_dict = torch.load(model_path,map_location='cuda:0')
    state_dict = torch.load(model_path)
    
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
        except:
            continue
        
    model.load_state_dict(own_state)
    print('reload model successfully!')
    return model


if __name__ == '__main__':
    import pickle
    from torch.utils.data import DataLoader
    DATA_PATH = '/gs/hs0/tgh-20IAA/jenn/data/VCTK_multispeaker_vcvqvae'
    with open(f'{DATA_PATH}/index.pkl', 'rb') as f:
        index = pickle.load(f)
    dataset = MultispeakerDataset(index, DATA_PATH)
    loader = DataLoader(dataset, batch_size=1)
    for x in loader:
        speaker_onehot, audio = x
        #print(f'x: {x}')
