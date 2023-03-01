from __future__ import division

# To include ham10000dataset
import sys

sys.path.insert(0, '../')

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ham10000dataset import ham10000_label_dict
from models.expert_model import MLPMixer


class synth_expert_hard_coded:
	
	def __init__(self, p_in=0.7, p_out=1/7, k=[3], device="cpu"):
		self.p_in = p_in
		self.p_out = p_out
		self.k = [ham10000_label_dict[i] for i in k]  # label to class number
		self.S = self.k # list : set of classes where the oracle predicts

		self.expert = MLPMixer(image_size=224, patch_size=16, in_channels=3, num_features=128, expansion_factor=2, num_layers=8, num_classes=7).to(device)
		self.expert.load_state_dict(torch.load('./MLP_Mixer_model/m_expert', map_location=device))
		self.expert.eval()
		self.n_classes = 7

	def MLPMixer(self, X, labels, Z):
		pred = self.expert(X, Z)
		# proxy for the expert's probability distribution
		conf = F.softmax(pred, dim=1)
		# sample from the expert's distribution
		c, outs = torch.max(conf, dim=1) #conf.multinomial(num_samples=1, replacement=True)
		return outs.cpu().tolist()

	# def predict_class_specific(self, input, labels, Z):
	# 	batch_size = labels.size()[0]  # batch_size
	# 	outs = [0] * batch_size
	# 	for i in range(0, batch_size):
	# 		if labels[i].item() in self.k:
	# 			outs[i] = labels[i].item()
	# 		else:
	# 			prediction_rand = random.randint(0, self.n_classes - 1)
	# 			outs[i] = prediction_rand
	# 	return outs

	# def predict(self, input, labels, Z):
	# 	batch_size = labels.size()[0]  # batch_size
	# 	outs = [0] * batch_size
	# 	for i in range(0, batch_size):
	# 		if labels[i].item() <= self.k:
	# 			outs[i] = labels[i].item()
	# 		else:
	# 			prediction_rand = random.randint(0, self.n_classes - 1)
	# 			outs[i] = prediction_rand
	# 	return outs

	def predict_prob(self, input, labels, Z):
		batch_size = labels.size()[0]
		outs = [0] * batch_size
		for i in range(0, batch_size):
			if labels[i].item() in self.k:
				coin_flip = np.random.binomial(1, self.p_in)
				if coin_flip == 1:
					outs[i] = labels[i].item()
				if coin_flip == 0:
					outs[i] = random.randint(0, self.n_classes - 1)
			else:
				coin_flip = np.random.binomial(1, self.p_out)
				if coin_flip == 1:
					outs[i] = labels[i].item()
				if coin_flip == 0:
					outs[i] = random.randint(0, self.n_classes - 1)
		return outs

	def predict_prob_ham10000_2(self, input, labels, Z):
		batch_size = labels.size()[0]
		outs = [0] * batch_size
		for i in range(0, batch_size):
			if labels[i].item() in self.S:
				coin_flip = np.random.binomial(1, self.p_in)
				if coin_flip == 1:
					outs[i] = labels[i].item()
				if coin_flip == 0:
					in_list = list(set(range(self.n_classes)) - set(self.S))
					if len(in_list) == 0:  # empty
						in_list = list(set(range(self.n_classes)) - set([labels[i].item()]))
					outs[i] = random.choice(in_list)
			else:
				coin_flip = np.random.binomial(1, self.p_out)
				if coin_flip == 1:
					outs[i] = labels[i].item()
				if coin_flip == 0:
					# Delete label which wasn't predicted!
					out_list = list(set(range(self.n_classes)) - set(self.S + [labels[i].item()]))
					outs[i] = random.choice(out_list)
		return outs

	def predict_random(self, input, labels, Z):
		batch_size = labels.size()[0]  # batch_size
		outs = [0] * batch_size
		for i in range(0, batch_size):
			prediction_rand = random.randint(0, self.n_classes - 1)
			outs[i] = prediction_rand
		return outs

class synth_expert:
	
	def __init__(self, device="cpu"):
		self.expert = MLPMixer(image_size=224, patch_size=16, in_channels=3, num_features=128, expansion_factor=2, num_layers=8, num_classes=7).to(device)
		self.expert.load_state_dict(torch.load('./MLP_Mixer_model/m_expert', map_location=device))
		self.expert.eval()
		self.n_classes = 7

	def MLPMixer(self, X, labels, Z):
		pred = self.expert(X, Z)
		# proxy for the expert's probability distribution
		conf = F.softmax(pred, dim=1)
		# sample from the expert's distribution
		c, outs = torch.max(conf, dim=1) #conf.multinomial(num_samples=1, replacement=True)
		return outs.cpu().tolist()

	def predict(self, input, labels, Z, k=3):
		batch_size = labels.size()[0]  # batch_size
		outs = [0] * batch_size
		for i in range(0, batch_size):
			if labels[i].item() <= k:
				outs[i] = labels[i].item()
			else:
				prediction_rand = random.randint(0, self.n_classes - 1)
				outs[i] = prediction_rand
		return outs

	def predict_prob(self, input, labels, Z, k=3, p_in=0.75, p_out=1/7):
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


	def predict_random(self, input, labels, Z):
		batch_size = labels.size()[0]  # batch_size
		outs = [0] * batch_size
		for i in range(0, batch_size):
			prediction_rand = random.randint(0, self.n_classes - 1)
			outs[i] = prediction_rand
		return outs
