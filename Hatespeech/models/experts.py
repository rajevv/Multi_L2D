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
	def __init__(self, flip_prob=0.30, p_in=0.75):
		self.n_classes = 2
		self.flip_prob = flip_prob
		self.p_in = p_in

	# human expert
	def HumanExpert(self, X, labels, hpred):
		batch_size = labels.size()[0]
		outs = [0] * batch_size

		for i in range(0, batch_size):
			outs[i] = hpred[i].item()

		return outs

	# flips human label with prob. flip_prob
	def FlipHuman(self, X, labels, hpred):
		batch_size = labels.size()[0]  # batch_size
		outs = [0] * batch_size
		for i in range(0, batch_size):
			coin_flip = np.random.binomial(1, self.flip_prob)
			if coin_flip == 1:
				outs[i] = ((1 - hpred[i]) > 0)*1
			else:
				outs[i] = hpred[i].item()
		return outs


	# takes human prediction with prob. p_in, otherwise predicts randomly
	def predict_prob(self, input, labels, hpred):
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
		batch_size = labels.size()[0]
		outs = [0] * batch_size
		for i in range(0, batch_size):
			prediction_rand = random.randint(0, self.n_classes - 1)
			outs[i] = prediction_rand
		return outs