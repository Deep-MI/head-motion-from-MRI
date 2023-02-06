"""
This file contains classes for keeping track of scores during epochs, training and evaluation.
It also wraps some of scipys stats score metrics
"""


from collections import Iterable

import torch

import numpy as np
from scipy.stats import spearmanr, pearsonr, wilcoxon
from sklearn.metrics import r2_score

from modenet.utils.functional import unsqueeze_to_dimension

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n  # TODO: adjust for loss
        self.count += n
        if self.count == 0:
            self.avg = 0
        else:
            self.avg = self.sum / self.count


class ScoreKeeper:
    """Stores network outputs and ground truth to calculate scores"""
    def __init__(self, denormalizer=lambda x : x):
        self.reset()
        self.denormalizer = denormalizer

    def reset(self):
        self.output = None
        self.target = None
        self.subjects = []


    def update(self, output, target, subject=None):
        
        #output = unsqueeze_to_dimension(output, 2)
        #target = unsqueeze_to_dimension(target, 2)

        assert(output.shape == target.shape), 'output and target must have the same shape, but have {} and {}'.format(output.shape, target.shape)

        # # if len(output.shape) == 2:
        # #     output = output.squeeze(1)

        # assert(len(output.shape) == 1)

        if subject is not None:
            if isinstance(subject, str):
                self.subjects.append(subject)
            elif isinstance(subject, Iterable):
                self.subjects.extend(subject)
            else:
                print('WARNING: couldnt add subject indentifier in score keeper - unknown data type')

        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        output = self.denormalizer(output)
        target = self.denormalizer(target)

        if type(self.output) != np.ndarray and type(self.target) != np.ndarray:
            self.output = output
            self.target = target
        else:
            self.output = np.concatenate((self.output,output),0)
            self.target = np.concatenate((self.target,target),0)



    def calculateR2(self):
        # we could do a manual implementation to make training and valdation r2 score comparable,
        # by using the same variance
        return r2_score(self.target, self.output)

    def calculateMeanDifference(self): # TODO: how to make these comparable for different values?
        size = self.target.size
        #acc = torch.dist(output, target).item()

        return np.abs(self.output - self.target).sum().item()/size

    def calculateWilcoxon(self):
        means_target = np.mean(self.output, axis=1)
        means_output = np.mean(self.target, axis=1)

        return wilcoxon(means_target, means_output)

    def calculatePearsonR(self):
        return pearsonr(self.output, self.target)

    def calculateSpearmanR(self):
        t, p = spearmanr(self.output, self.target)
        return np.array(t)
