import math
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F



class synth_expert:
	'''
	simple class to describe our synthetic expert on CIFAR-10
	----
	k: number of classes expert can predict
	n_classes: number of classes (10+1 for CIFAR-10)
	'''
	def __init__(self):
		self.expert = MLPMixer(image_size=224, patch_size=16, in_channels=3, num_features=128, expansion_factor=2, num_layers=8, num_classes=7).to(device)
		self.expert.load_state_dict(torch.load('./Models/m_expert', map_location=device))
		self.expert.eval()

	def predict(self, X, labels, Z):
		pred = self.expert(X, Z)
		# proxy for the expert's probability distribution
		conf = F.softmax(pred, dim=1)
		# sample from the expert's distribution
		c, outs = torch.max(conf, dim=1) #conf.multinomial(num_samples=1, replacement=True)
		return outs.cpu().tolist()