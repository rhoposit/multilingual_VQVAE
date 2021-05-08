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
from typing import Tuple


class GradientReversalFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_forward: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(scale)
        return input_forward

    @staticmethod
    def backward(ctx, grad_backward: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, = ctx.saved_tensors
        return scale * -grad_backward, None



class GradientReversal(nn.Module):

    def __init__(self, scale: float):
        super(GradientReversal, self).__init__()
        self.scale = torch.tensor(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.scale)
