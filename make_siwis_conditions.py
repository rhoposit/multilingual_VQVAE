# ==================================================================================================
# Copyright (c) 2021, Jennifer Williams and Yamagishi Laboratory, National Institute of Informatics
# Author: Jennifer Williams (j.williams@ed.ac.uk)
# All rights reserved.
# ==================================================================================================


import math, pickle, os, sys
from collections import defaultdict


infile = "/path/to/data/SIWIS/wav_norm_vqvae/index.pkl"

def getstats(index):
    text = set()
    T = []
    LANG_utts = defaultdict(list)
    LANG_speakers = defaultdict(set)
    TXT_speakers = defaultdict(set)
    index = list(index.values())
    for speakerlist in index:
        for utt in speakerlist:
            lang = utt.split("_")[0]
            spk = utt.split("_")[2]
            txt = utt.split("_")[1]+"_"+utt.split("_")[3]
            text.add(txt)
            T.append(txt)
            LANG_speakers[lang].add(spk)
            TXT_speakers[txt].add(spk)
            LANG_utts[lang].append(utt)
#    print("\nSpeakers per language")
#    [print(k,sorted(v)) for k,v in LANG_speakers.items()]
#    print("Number of speakers per language")
#    [print(k, len(v)) for k,v in LANG_speakers.items()]
#    print("Number of utterances per language")
#    [print(k, len(v)) for k,v in LANG_utts.items()]
#    print("Number of utterances")
#    print("uniq", len(list(text)))
#    print("all", len(T))
    return text, T


def get_condition4_shared_utts(target_texts, index):
    hold_index = defaultdict(set)
    for speakerlist in list(index.values()):
        for utt in speakerlist:
            lang = utt.split("_")[0]
            spk = utt.split("_")[2]
            txt = utt.split("_")[1][0]+"_"+utt.split("_")[3]
            hold_index[spk].add(txt)
    list_of_sets = list(hold_index.values())
    shared = list_of_sets[0].intersection(*list_of_sets)
    return shared



def get_condition(target_texts, index):
    condition = []
    for speakerlist in list(index.values()):
        for utt in speakerlist:
            lang = utt.split("_")[0]
            spk = utt.split("_")[2]
            txt = utt.split("_")[1][0]+"_"+utt.split("_")[3]
            if txt in target_texts:
                condition.append(utt)
    return condition



def refine_condition(c_texts):
    hold_index = defaultdict(set)
    selected = defaultdict(list)
    selected_txt = []
    for utt in c_texts:
        lang = utt.split("_")[0]
        spk = utt.split("_")[2]
        txt = utt.split("_")[1][0]+"_"+utt.split("_")[3]
        hold_index[spk].add(txt)
    list_of_sets = list(hold_index.values())
    shared = list(list_of_sets[0].intersection(*list_of_sets))[:10]
    for utt in c_texts:
        lang = utt.split("_")[0]
        spk = utt.split("_")[2]
        txt = utt.split("_")[1][0]+"_"+utt.split("_")[3]
        if txt in shared:
            if len(selected[spk]) < 10:
                selected[spk].append(utt)
                selected_txt.append(txt)
    return selected, selected_txt


def refine_condition1(index, spklist):
    selected = defaultdict(list)
    selected_txt = []
    for speakerlist in list(index.values()):
        for utt in speakerlist:
            lang = utt.split("_")[0]
            spk = utt.split("_")[2]
            txt = utt.split("_")[1][0]+"_"+utt.split("_")[3]
            if spk in spklist:
                if len(selected[spk]) < 10:
                    selected[spk].append(utt)
                    selected_txt.append(txt)
    return selected, selected_txt




with open(infile, 'rb') as f:
    index = pickle.load(f)
# M = 03, 01, 21
# F = 08, 18, 05
heldout_speakers = ['08', '18', '05', '03', '01', '21']
# seen speakers: 04F, 06F, 07F, 14M, 13M, 20M
seen_speakers = ['04', '06', '07', '14', '13', '20']
index_heldout = defaultdict(list)
index_c2 = defaultdict(list)
for spkset in index:
    for item in spkset:
        spk = item.split("_")[2]
        if spk in heldout_speakers:
            index_heldout[spk].append(item)
        if spk in seen_speakers:
            if len(index_c2[spk]) < 10:
                index_c2[spk].append(item)
htextset, htextlist = getstats(index_heldout)
c2textset, c2textlist = getstats(index_c2)

# texts and speakers never seen in training
condition4_text = get_condition4_shared_utts(htextset, index_heldout)
c4 = get_condition(condition4_text, index_heldout)
print("condition4 texts:", len(c4))
c4, c4txt = refine_condition(c4)
[print(k,len(v)) for k,v in c4.items()]


index_train = defaultdict(list)
index_c1 = defaultdict(list)
for spkset in index:
    for item in spkset:
        spk = item.split("_")[2]
        txt = item.split("_")[1][0]+"_"+item.split("_")[3]
        txt2 = item.split("_")[1]+"_"+item.split("_")[3]
        if spk not in heldout_speakers:
            if txt not in c4txt:
                if txt2 not in c2textset:
                    index_train[spk].append(item)
            

ttextset, ttextlist = getstats(index_train)
condition3_text = ttextset.intersection(htextset)

# text seen during training, by speakers who were unseen
#print("unique condition3 texts", len(list(condition3_text)))
c3 = get_condition(condition3_text, index_heldout)
print("condition3 texts:", len(c3))
c3, c3txt = refine_condition(c3)
[print(k,len(v)) for k,v in c3.items()]

# speakers that were seen in training, but the texts were not
c2 = index_c2
num = sum([len(v) for k,v in c2.items()])
print("condition2 texts:", num)
[print(k,len(v)) for k,v in c2.items()]


# texts that were seen in training, and so were the speakers
c1, c1txt = refine_condition1(index_train, seen_speakers)
num = sum([len(v) for k,v in c1.items()])
print("condition1 texts:", num)
[print(k,len(v)) for k,v in c1.items()]

# condition1: seen speakers, seen text ==> training set
# condition2: seen speakers, unseen text ==> validation set
# condition3: unseen speakers, seen text
# condition4: unseen speakers, unseen text
outfile = "/home/s1738075/VQ_experiments/test_lists/siwis_condition1.txt"
output = open(outfile, "w")
[output.write("\n".join(v)+"\n") for k,v in c1.items()]
output.close()
outfile = "/home/s1738075/VQ_experiments/test_lists/siwis_condition2.txt"
output = open(outfile, "w")
[output.write("\n".join(v)+"\n") for k,v in c2.items()]
output.close()
outfile = "/home/s1738075/VQ_experiments/test_lists/siwis_condition3.txt"
output = open(outfile, "w")
[output.write("\n".join(v)+"\n") for k,v in c3.items()]
output.close()
outfile = "/home/s1738075/VQ_experiments/test_lists/siwis_condition4.txt"
output = open(outfile, "w")
[output.write("\n".join(v)+"\n") for k,v in c4.items()]
output.close()


a = [v for k,v in index_train.items()]
newpickle = "/home/s1738075/data/SIWIS/wav_norm_vqvae/index2.pkl"
with open(newpickle, 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
#newpickle = "/home/s1738075/data/SIWIS/wav_norm_vqvaef0/index2.pkl"
#with open(newpickle, 'wb') as handle:
#    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)




with open('/home/s1738075/data/SIWIS/wav_norm_vqvae/index2.pkl', 'rb') as f:
    index = pickle.load(f)
train_set = index[:]
test_index = [x[:75] for i, x in enumerate(train_set)]
train_index = [x[75:] for i, x in enumerate(train_set)]
print("Train set speakers: ", len(train_index))
print("Train set total: ", sum( [ len(listElem) for listElem in train_index]))
print("Test set speakers: ", len(test_index))
print("Test set total: ", sum( [ len(listElem) for listElem in test_index]))
