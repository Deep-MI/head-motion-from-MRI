"""
loss functions for regression tasks
"""


import torch
import torch.nn as nn



class L1L2Loss:
    def __init__(self, weight=1):
        self.mse = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss() # Mean absolute error
        self.weight = weight

    def __call__(self, output, target):
        return self.mse(output, target) + self.l1(output, target) * self.weight

class KLDivLoss:
    """Returns K-L Divergence loss as proposed by Peng et al. 2021 for brain age predicition 
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    def __init__(self):
    	self.loss_func = nn.KLDivLoss(reduction='sum')

    def __call__(self, x, y):
    	y += 1e-16
    	n = y.shape[0]
    	loss = self.loss_func(x, y) / n
    	return loss
