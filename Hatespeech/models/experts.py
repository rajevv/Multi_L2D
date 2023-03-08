from __future__ import division

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class synth_expert:
    """
    Args:
        flip_prob (float): probability of flipping human label
        p_in (float): probability of taking human label
    """
    def __init__(self, flip_prob=0.30, p_in=0.75):
        self.n_classes = 3
        self.flip_prob = flip_prob
        self.p_in = p_in

    # human expert
    def HumanExpert(self, X, labels, hpred):
        """Expert probability prediction using real human labels.
        Args:
            X: input data
            labels: real human labels
            hpred: human prediction

        Returns:
            outs: human expert predictions
        """
        batch_size = labels.size()[0]
        outs = [0] * batch_size

        for i in range(0, batch_size):
            outs[i] = hpred[i].item()

        return outs

    def FlipHuman(self, X, labels, hpred):
        """Flip human label with probability flip_prob.
        Args:
            X: input data
            labels: real human labels
            hpred: human prediction
        Returns:
            outs: human expert predictions
                """
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            coin_flip = np.random.binomial(1, self.flip_prob)
            if coin_flip == 1:
                outs[i] = ((1 - hpred[i]) > 0)*1
            else:
                outs[i] = hpred[i].item()
        return outs

    def IncorrectExpert(self, X, labels, hpred):
        """Expert probability prediction using incorrect labels.
        Args:
            X: input data
            labels: real human labels
            hpred: human prediction
        Returns:
            outs: human expert predictions
        """
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            outs[i] = ((1 - labels[i]) > 0) * 1
        return outs

    # takes human prediction with prob. p_in, otherwise predicts randomly

    def predict_prob(self, input, labels, hpred):
        """Expert predict with probability p_in.
        Args:
            X: input data
            labels: real human labels
            hpred: human prediction

        Returns:        
            outs: human expert predictions
        """
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            coin_flip = np.random.binomial(1, self.p_in)
            if coin_flip == 1:
                outs[i] = hpred[i].item()
            if coin_flip == 0:
                outs[i] = random.randint(0, self.n_classes - 1)
        return outs

    # predicts randomly
    def predict_random(self, input, labels, hpred):
        """Expert predict randomly.
        Args:
            X: input data
            labels: real human labels
            hpred: human prediction

        Returns:
            outs: human expert predictions
        """
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            prediction_rand = random.randint(0, self.n_classes - 1)
            outs[i] = prediction_rand
        return outs
