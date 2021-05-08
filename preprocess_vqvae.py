# ==================================================================================================
# Copyright (c) 2021, Jennifer Williams and Yamagishi Laboratory, National Institute of Informatics
# Author: Jennifer Williams (j.williams@ed.ac.uk)
# All rights reserved.
# ==================================================================================================


import sys
import glob
import pickle
import os
import multiprocessing as mp
from utils.dsp import *

import sys
import glob
import pickle
import os
import multiprocessing as mp
from utils.dsp import *

SEG_PATH = "/gs/hs0/tgh-20IAA/jenn/data/VCTK-0.92_wav16_norm"
DATA_PATH = "/gs/hs0/tgh-20IAA/jenn/data/VCTK-0.92_vcvqvae_norm/"




def get_files(path):
    next_speaker_id = 0
    speaker_ids = {}
    filenames = []
    for filename in glob.iglob(f'{path}/*.wav', recursive=True):
        speaker_name = filename.split('/')[-1].split("_")[0]
        if speaker_name not in speaker_ids:
            speaker_ids[speaker_name] = next_speaker_id
            next_speaker_id += 1
            filenames.append([])
        filenames[speaker_ids[speaker_name]].append(filename)

    return filenames, speaker_ids

files, spks = get_files(SEG_PATH)

def process_file(i, path):
    dir = f'{DATA_PATH}/'
    name = path.split('/')[-1][:-4] # Drop .wav
    filename = f'{dir}/{name}.npy'
    if os.path.exists(filename):
        print(f'{filename} already exists, skipping')
        return
    floats = load_wav(path, encode=False)
    trimmed, _ = librosa.effects.trim(floats, top_db=25)
    quant = (trimmed * (2**15 - 0.5) - 0.5).astype(np.int16)
    if max(abs(quant)) < 2048:
        print(f'audio fragment too quiet ({max(abs(quant))}), skipping: {path}')
        return
    if len(quant) < 10000:
        print(f'audio fragment too short ({len(quant)} samples), skipping: {path}')
        return
    os.makedirs(dir, exist_ok=True)
    np.save(filename, quant)
    return name

index = []
with mp.Pool(8) as pool:
    for i, speaker in enumerate(files):
        res = pool.starmap_async(process_file, [(i, path) for path in speaker]).get()
        index.append([x for x in res if x])
        print(f'Done processing speaker {i}')

os.makedirs(DATA_PATH, exist_ok=True)
with open(f'{DATA_PATH}/index.pkl', 'wb') as f:
    pickle.dump(index, f)


spk_map = "spkmap_vqvae.txt"
output = open(spk_map, "w")
for k,v in spks.items():
    output.write(str(k)+","+str(v)+"\n")
output.close()
