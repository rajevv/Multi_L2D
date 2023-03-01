import argparse
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable





class Classifier(nn.Module):
	def __init__(self, out_size=None):
		super(Classifier, self).__init__()
		self.resnet34 = torchvision.models.resnet34(pretrained=True)
		num_ftrs = self.resnet34.fc.in_features
		self.resnet34.fc = nn.Sequential(
			nn.Linear(num_ftrs, out_size))
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		x = self.resnet34(x)
		return x #self.softmax(x)


class Deferrer(nn.Module):
	def __init__(self, out_size=None):
		super(Deferrer, self).__init__()
		self.resnet34 = torchvision.models.resnet34(pretrained=True)
		num_ftrs = self.resnet34.fc.in_features
		self.resnet34.fc = nn.Sequential(
			nn.Linear(num_ftrs, out_size))
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.resnet34(x)
		return self.sigmoid(x)