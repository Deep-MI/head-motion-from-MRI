"""
Pytorch transforms for data augmentation
"""


import torch
import os

import numpy as np
import random

class RandomFlip(object):
    """
    Reverse the order of elements in the image along the given axes.
    """
    def __init__(self, axes=1, flip_probability=0.5, p=1, keys=None):
        self.flip_probability = flip_probability

    def apply_transform(self, sample):
        if p == 1 or random.random() < p:
            sample['image'] = torch.flip(sample['image'], axis=self.axes +2) # add two to ignore batch and channel dimensions



class RandomIntensityScaling(object):
    """
    Scale image intensities by a random value
    """
    def __init__(self, range=(0.9,1.1)):
        self.range = range

    def __call__(self, sample):
        sample['image'] = torch.FloatTensor(1).uniform_(0.9, 1.1) * sample['image']
        return {**sample}
