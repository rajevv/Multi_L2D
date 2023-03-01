from __future__ import division
import math
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
from expert_model import MLPMixer
from data_utils import ham10000_label_dict

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
	
	def __init__(self):
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





# class synth_expert:
# 	'''
# 	simple class to describe our synthetic expert on CIFAR-10
# 	----
# 	k: number of classes expert can predict
# 	n_classes: number of classes (10+1 for CIFAR-10)
# 	'''
# 	def __init__(self, k, n_classes, p_in=1, p_out=0.2):
# 		self.k = k
# 		self.n_classes = n_classes
# 		self.p_in = p_in
# 		self.p_out = p_out if p_out is not None else 1/self.n_classes

# 	def predict(self, input, labels):
# 		batch_size = labels.size()[0]  # batch_size
# 		outs = [0] * batch_size
# 		for i in range(0, batch_size):
# 			if labels[i][0].item() <= self.k:
# 				outs[i] = labels[i][0].item()
# 			else:
# 				prediction_rand = random.randint(0, self.n_classes - 1)
# 				outs[i] = prediction_rand
# 		return outs

# 	def predict(self, input, labels):
# 		batch_size = labels.size()[0]  # batch_size
# 		outs = [0] * batch_size
# 		for i in range(0, batch_size):
# 			if labels[i][0].item() <= self.k:
# 				outs[i] = labels[i][0].item()
# 			else:
# 				prediction_rand = random.randint(0, self.n_classes - 1)
# 				outs[i] = prediction_rand
# 		return outs

# 	def predict_biasedK(self, input, labels):
# 		batch_size = labels.size()[0]  # batch_size
# 		outs = [0] * batch_size
# 		for i in range(0, batch_size):
# 			if labels[i][0].item() <= self.k:
# 				coin_flip = np.random.binomial(1, 0.7)
# 				if coin_flip == 1:
# 					outs[i] = labels[i][0].item()
# 				if coin_flip == 0:
# 					outs[i] = random.randint(0, self.n_classes - 1)
# 			else:
# 				prediction_rand = random.randint(0, self.n_classes - 1)
# 				outs[i] = prediction_rand
# 		return outs

# 	def predict_prob_cifar(self, input, labels):
# 		batch_size = labels.size()[0]
# 		outs = [0] * batch_size
# 		for i in range(0, batch_size):
# 			if labels[i][0].item() <= self.k:
# 				coin_flip = np.random.binomial(1, self.p_in)
# 				if coin_flip == 1:
# 					outs[i] = labels[i][0].item()
# 				if coin_flip == 0:
# 					outs[i] = random.randint(0, self.n_classes - 1)
# 			else:
# 				coin_flip = np.random.binomial(1, self.p_out)
# 				if coin_flip == 1:
# 					outs[i] = labels[i][0].item()
# 				if coin_flip == 0:
# 					outs[i] = random.randint(0, self.n_classes - 1)
# 		return outs

# 	def predict_prob(self, input, labels, p1=0.75, p2=0.20):
# 		batch_size = labels.size()[0]
# 		outs = [0] * batch_size
# 		for i in range(0, batch_size):
# 			if labels[i][0].item() <= self.k:
# 				coin_flip = np.random.binomial(1, p1)
# 				if coin_flip == 1:
# 					outs[i] = labels[i][0].item()
# 				if coin_flip == 0:
# 					outs[i] = random.randint(0, self.n_classes - 1)
# 			else:
# 				coin_flip = np.random.binomial(1, p2)
# 				if coin_flip == 1:
# 					outs[i] = labels[i][0].item()
# 				if coin_flip == 0:
# 					outs[i] = random.randint(0, self.n_classes - 1)
# 		return outs



# 	def predict_biased(self, input, labels):
# 		batch_size = labels.size()[0]
# 		outs = [0]*batch_size
# 		for i in range(0, batch_size):
# 			coin_flip = np.random.binomial(1, 0.7)
# 			if coin_flip == 1:
# 				outs[i] = labels[i][0].item()
# 			if coin_flip == 0:
# 				outs[i] = random.randint(0, self.n_classes - 1)
# 		return outs

# 	def predict_random(self, input, labels):
# 		batch_size = labels.size()[0]  # batch_size
# 		outs = [0] * batch_size
# 		for i in range(0, batch_size):
# 			prediction_rand = random.randint(0, self.n_classes - 1)
# 			outs[i] = prediction_rand
# 		return outs

# 	#expert which only knows k labels and even outside of K predicts randomly from K
# 	def predict_severe(self, input, labels):
# 		batch_size = labels.size()[0]  # batch_size
# 		outs = [0] * batch_size
# 		for i in range(0, batch_size):
# 			if labels[i][0].item() <= self.k:
# 				outs[i] = labels[i][0].item()
# 			else:
# 				prediction_rand = random.randint(0, self.k)
# 				outs[i] = prediction_rand
# 		return outs

# 	#when the input is OOD, expert predicts correctly else not
# 	def oracle(self, input, labels):
# 		batch_size = labels.size()[0]
# 		outs = [0]*batch_size
# 		for i in range(0, batch_size):
# 			if labels[i][1].item() == 0:
# 				outs[i] = labels[i][0]
# 			else:
# 				if labels[i][0].item() <= self.k:
# 					outs[i] = labels[i][0].item()
# 				else:
# 					prediction_rand = random.randint(0, self.n_classes - 1)
# 					outs[i] = prediction_rand
# 		return outs