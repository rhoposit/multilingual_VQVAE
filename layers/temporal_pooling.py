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

class TemporalPooling_phn(nn.Module) :
    """
    Input: (N, samples, channels) numeric tensor
    Output: (N, 1, n_channels) numeric tensor 
    """
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels        
        # this is size of sample outputs from encoder
        self.avg = nn.AvgPool1d(101)
        self.fc1 = nn.Linear(n_channels, n_channels)
        self.fc2 = nn.Linear(n_channels, n_channels)
        self.fc3 = nn.Linear(n_channels, n_channels)
        self.fc4 = nn.Linear(n_channels, n_channels)
        
        return

    def forward(self, x):
        x = x.permute(0, 2, 1)
        tap = self.avg(x)
        tap = tap.squeeze(2)
        ff1 = self.fc1(tap)
        ff2 = self.fc2(ff1)
        ff3 = self.fc3(ff2)
        ff4 = self.fc4(ff3)
        ff4 = ff4.unsqueeze(2)
        ret = ff4.permute(0, 2, 1)
        return ret




class TemporalPooling_spk(nn.Module) :
    """
    Input: (N, samples, channels) numeric tensor
    Output: (N, 1, n_channels) numeric tensor 
    """
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels        
        # this is size of sample outputs from encoder
        self.avg = nn.AvgPool1d(101)
        self.fc1 = nn.Linear(n_channels, n_channels)
        self.fc2 = nn.Linear(n_channels, n_channels)        
        return

    def forward(self, x):
        x = x.permute(0, 2, 1)
        tap = self.avg(x)
        tap = tap.squeeze(2)
        ff1 = self.fc1(tap)
        ff2 = self.fc2(ff1)
        ff2 = ff2.unsqueeze(2)
        ret = ff2.permute(0, 2, 1)
        return ret

   
