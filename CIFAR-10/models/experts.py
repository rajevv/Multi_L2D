import numpy as np
import torch
import torch.nn as nn
import random


class synth_expert2:
	def __init__(self, k1, k2, n_classes):
		self.k1 = k1
		self.k2 = k2
		self.n_classes = n_classes
		
	def predict(self, input, labels):
		batch_size = labels.size()[0]  # batch_size
		outs = [0] * batch_size
		for i in range(0, batch_size):
			if labels[i][0].item() < self.k2 and labels[i][0].item() >= self.k1:
				outs[i] = labels[i][0].item()
			else:
				prediction_rand = random.randint(0, self.n_classes - 1)
				outs[i] = prediction_rand
		return outs



class synth_expert:
	'''
	simple class to describe our synthetic expert on CIFAR-10
	----
	k: number of classes expert can predict
	n_classes: number of classes (10+1 for CIFAR-10)
	'''
	def __init__(self, k, n_classes, p_in=1, p_out=0.2):
		self.k = k
		self.n_classes = n_classes
		self.p_in = p_in
		self.p_out = p_out if p_out is not None else 1/self.n_classes

	def predict(self, input, labels):
		batch_size = labels.size()[0]  # batch_size
		outs = [0] * batch_size
		for i in range(0, batch_size):
			if labels[i][0].item() <= self.k:
				outs[i] = labels[i][0].item()
			else:
				prediction_rand = random.randint(0, self.n_classes - 1)
				outs[i] = prediction_rand
		return outs

	def predict(self, input, labels):
		batch_size = labels.size()[0]  # batch_size
		outs = [0] * batch_size
		for i in range(0, batch_size):
			if labels[i][0].item() <= self.k:
				outs[i] = labels[i][0].item()
			else:
				prediction_rand = random.randint(0, self.n_classes - 1)
				outs[i] = prediction_rand
		return outs

	def predict_biasedK(self, input, labels):
		batch_size = labels.size()[0]  # batch_size
		outs = [0] * batch_size
		for i in range(0, batch_size):
			if labels[i][0].item() <= self.k:
				coin_flip = np.random.binomial(1, 0.7)
				if coin_flip == 1:
					outs[i] = labels[i][0].item()
				if coin_flip == 0:
					outs[i] = random.randint(0, self.n_classes - 1)
			else:
				prediction_rand = random.randint(0, self.n_classes - 1)
				outs[i] = prediction_rand
		return outs

	def predict_prob_cifar(self, input, labels):
		batch_size = labels.size()[0]
		outs = [0] * batch_size
		for i in range(0, batch_size):
			if labels[i][0].item() <= self.k:
				coin_flip = np.random.binomial(1, self.p_in)
				if coin_flip == 1:
					outs[i] = labels[i][0].item()
				if coin_flip == 0:
					outs[i] = random.randint(0, self.n_classes - 1)
			else:
				coin_flip = np.random.binomial(1, self.p_out)
				if coin_flip == 1:
					outs[i] = labels[i][0].item()
				if coin_flip == 0:
					outs[i] = random.randint(0, self.n_classes - 1)
		return outs

	def predict_prob(self, input, labels, p1=0.75, p2=0.20):
		batch_size = labels.size()[0]
		outs = [0] * batch_size
		for i in range(0, batch_size):
			if labels[i][0].item() <= self.k:
				coin_flip = np.random.binomial(1, p1)
				if coin_flip == 1:
					outs[i] = labels[i][0].item()
				if coin_flip == 0:
					outs[i] = random.randint(0, self.n_classes - 1)
			else:
				coin_flip = np.random.binomial(1, p2)
				if coin_flip == 1:
					outs[i] = labels[i][0].item()
				if coin_flip == 0:
					outs[i] = random.randint(0, self.n_classes - 1)
		return outs



	def predict_biased(self, input, labels):
		batch_size = labels.size()[0]
		outs = [0]*batch_size
		for i in range(0, batch_size):
			coin_flip = np.random.binomial(1, 0.7)
			if coin_flip == 1:
				outs[i] = labels[i][0].item()
			if coin_flip == 0:
				outs[i] = random.randint(0, self.n_classes - 1)
		return outs

	def predict_random(self, input, labels):
		batch_size = labels.size()[0]  # batch_size
		outs = [0] * batch_size
		for i in range(0, batch_size):
			prediction_rand = random.randint(0, self.n_classes - 1)
			outs[i] = prediction_rand
		return outs

	#expert which only knows k labels and even outside of K predicts randomly from K
	def predict_severe(self, input, labels):
		batch_size = labels.size()[0]  # batch_size
		outs = [0] * batch_size
		for i in range(0, batch_size):
			if labels[i][0].item() <= self.k:
				outs[i] = labels[i][0].item()
			else:
				prediction_rand = random.randint(0, self.k)
				outs[i] = prediction_rand
		return outs

	#when the input is OOD, expert predicts correctly else not
	def oracle(self, input, labels):
		batch_size = labels.size()[0]
		outs = [0]*batch_size
		for i in range(0, batch_size):
			if labels[i][1].item() == 0:
				outs[i] = labels[i][0]
			else:
				if labels[i][0].item() <= self.k:
					outs[i] = labels[i][0].item()
				else:
					prediction_rand = random.randint(0, self.n_classes - 1)
					outs[i] = prediction_rand
		return outs
