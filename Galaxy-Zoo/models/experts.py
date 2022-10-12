from __future__ import division
import math
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class synth_expert:
	'''
	simple class to describe our synthetic expert on CIFAR-10
	----
	k: number of classes expert can predict
	n_classes: number of classes (10+1 for CIFAR-10)
	'''
	def __init__(self):
		self.n_classes = 2

	def HumanExpert(self, X, labels, hpred):
		return hpred.cpu().tolist()

	# Expert flips human prediction with probability p_in
	def FlipHuman(self, input, labels, hpred, p_in=0.30):
		batch_size = labels.size()[0]  # batch_size
		outs = [0] * batch_size
		for i in range(0, batch_size):
			coin_flip = np.random.binomial(1, p_in)
			if coin_flip == 1:
				outs[i] = ((1 - hpred[i]) > 0)*1
			else:
				outs[i] = hpred[i]
		return outs

	# Experts predict correctly for some class, not perfectly other class
	def predict_prob(self, input, labels, hpred, k=0, p_in=0.75, p_out=1/2):
		batch_size = labels.size()[0]
		outs = [0] * batch_size
		for i in range(0, batch_size):
			if labels[i].item() <= k:
				coin_flip = np.random.binomial(1, p_in)
				if coin_flip == 1:
					outs[i] = labels[i].item()
				if coin_flip == 0:
					outs[i] = random.randint(0, self.n_classes - 1)
			else:
				coin_flip = np.random.binomial(1, p_out)
				if coin_flip == 1:
					outs[i] = labels[i].item()
				if coin_flip == 0:
					outs[i] = random.randint(0, self.n_classes - 1)
		return outs


	# completely random
	def predict_random(self, input, labels, hpred):
		batch_size = labels.size()[0]  # batch_size
		outs = [0] * batch_size
		for i in range(0, batch_size):
			prediction_rand = random.randint(0, self.n_classes - 1)
			outs[i] = prediction_rand
		return outs