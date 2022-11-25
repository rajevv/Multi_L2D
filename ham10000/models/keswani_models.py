import math
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import argparse
import os
import shutil
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
# from expert_model import MLPMixer
# from data_utils import *
# from models.resnet34 import *
# from models.experts import *
# from losses.losses import *




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