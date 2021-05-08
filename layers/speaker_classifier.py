# ==================================================================================================
# Copyright (c) 2021, Jennifer Williams and Yamagishi Laboratory, National Institute of Informatics
# Author: Jennifer Williams (j.williams@ed.ac.uk)
# All rights reserved.
# ==================================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils.logger as logger

class SpeakerClassifier(nn.Module) :
    def __init__(self, indims, n_spk):
        super().__init__()
        self.ff = nn.Linear(indims, n_spk)
        return

    
    def forward(self, x):
        predicted_speaker = self.ff(x)
        return predicted_speaker
