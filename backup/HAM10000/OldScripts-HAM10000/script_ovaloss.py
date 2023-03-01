from __future__ import division
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
from expert_model import MLPMixer
import wandb



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
class ResNet34_defer(nn.Module):
	def __init__(self, out_size):
		super(ResNet34_defer, self).__init__()
		self.resnet34 = torchvision.models.resnet34(pretrained=True)
		num_ftrs = self.resnet34.fc.in_features
		self.resnet34.fc = nn.Sequential(
			nn.Linear(num_ftrs, out_size))
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.resnet34(x)
		return x, self.sigmoid(x)

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
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res




def metrics_print(net,expert_fn, n_classes, loader):
	'''
	Computes metrics for deferal
	-----
	Arguments:
	net: model
	expert_fn: expert model
	n_classes: number of classes
	loader: data loader
	'''
	correct = 0
	correct_sys = 0
	exp = 0
	exp_total = 0
	total = 0
	real_total = 0
	alone_correct = 0
	losses = []
	with torch.no_grad():
		for data in loader:
			images, labels, Z = data
			images, labels, Z = images.to(device), labels.to(device), Z.float().to(device)
			outputs,_ = net(images)
			conf, predicted = torch.max(outputs.data, 1)
			batch_size = outputs.size()[0]  # batch_size
			exp_prediction = expert_fn(images, labels, Z)
			m2 = [0] * batch_size
			for j in range(0, batch_size):
				if exp_prediction[j] == labels[j].item():
					m2[j] = 1
				else:
					m2[j] = 0
			m2 = torch.tensor(m2)
			m2 = m2.to(device)

			m3 = [0] * batch_size

			for i, j in enumerate(m2):
				if j == 1:
					m3[i] = alpha
				if j == 0:
					m3[i] = 1

			m3 = torch.tensor(m3)
			m3 = m3.to(device)

			losses.append(OVAloss(outputs, labels, m2, m3, n_classes).item())
			for i in range(0, batch_size):
				r = (predicted[i].item() == n_classes) #the max is rejetion class
				prediction = predicted[i]
				if predicted[i] == n_classes:
					max_idx = 0
					# get second max
					for j in range(0, n_classes):
						if outputs.data[i][j] >= outputs.data[i][max_idx]:
							max_idx = j
					prediction = max_idx
				else:
					prediction = predicted[i]
				alone_correct += (prediction == labels[i]).item()
				if r == 0:
					total += 1
					correct += (predicted[i] == labels[i]).item()
					correct_sys += (predicted[i] == labels[i]).item()
				if r == 1:
					exp += (exp_prediction[i] == labels[i].item())
					correct_sys += (exp_prediction[i] == labels[i].item())
					exp_total += 1
				real_total += 1
	cov = str(total) + str(" out of") + str(real_total)
	to_print = {"coverage": cov, "system_accuracy": 100 * correct_sys / real_total,
				"expert_accuracy": 100 * exp / (exp_total + 0.0002),
				"classifier_accuracy": 100 * correct / (total + 0.0001),
				"alone_classifier": 100 * alone_correct / real_total,
				"validation_loss": np.average(losses)}
	print(to_print, flush=True)
	return to_print


def OVAloss(outputs, labels, m2, m3, n_classes):
	batch_size = outputs.size()[0]
	l1 = LogisticLoss(outputs[range(batch_size), labels], 1)
	l2 = torch.sum(LogisticLoss(outputs[:,:n_classes], -1), dim=1) - LogisticLoss(outputs[range(batch_size),labels],-1)
	l3 = LogisticLoss(outputs[range(batch_size), n_classes], -1)

	l4 = LogisticLoss(outputs[range(batch_size), n_classes], 1)

	l5 = m2*(l4 - l3)

	l = m3*(l1 + l2) + l3 + l5
	return torch.mean(l)


def LogisticLoss(outputs, y):
	outputs[torch.where(outputs==0.0)] = (-1*y)*(-1*np.inf)
	l = torch.log2(1 + torch.exp((-1*y)*outputs))
	return l


def train_reject(iters, warmup_iters, lrate, train_loader, model, optimizer, scheduler, epoch, expert_fn, n_classes, alpha):
	"""Train for one epoch on the training set with deferral"""
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	epoch_train_loss = []
	for i, (input, target, Z) in enumerate(train_loader):
		if iters < warmup_iters:
				lr = lrate*float(iters) / warmup_iters
				print(iters, lr)
				for param_group in optimizer.param_groups:
					param_group['lr'] = lr

		target = target.to(device)
		input = input.to(device)
		Z = Z.float().to(device)

		# compute output
		output,_ = model(input)

		# get expert  predictions and costs
		batch_size = output.size()[0]  # batch_size
		m = expert_fn(input, target, Z)
		m2 = [0] * batch_size
		for j in range(0, batch_size):
			if m[j] == target[j].item():
				m2[j] = 1
			else:
				m2[j] = 0
		#m = torch.tensor(m)
		m2 = torch.tensor(m2)
		#m = m.to(device)
		m2 = m2.to(device)

		m3 = [0] * batch_size
		for i, j in enumerate(m2):
			if j == 1:
				m3[i] = alpha
			if j == 0:
				m3[i] = 0

		m3 = torch.tensor(m3)
		m3 = m3.to(device)

		# done getting expert predictions and costs 
		# compute loss
		#criterion = nn.CrossEntropyLoss()
		loss = OVAloss(output, target, m2, m3, n_classes)
		epoch_train_loss.append(loss.item())

		# measure accuracy and record loss
		prec1 = accuracy(output.data, target, topk=(1,))[0]
		losses.update(loss.data.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if not iters < warmup_iters:
			scheduler.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()
		iters+=1

		if i % 10 == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
				epoch, i, len(train_loader), batch_time=batch_time,
				loss=losses, top1=top1), flush=True)
	return iters, np.average(epoch_train_loss)

def run_reject(model, train_dataset, valid_dataset, n_dataset, expert_fn, epochs, alpha, batch_size, save_path='./'):

	

	kwargs = {'num_workers': 0, 'pin_memory': True}

	train_loader = torch.utils.data.DataLoader(train_dataset,
											   batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(valid_dataset,
											   batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
	# get the number of model parameters
	print('Number of model parameters: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])), flush=True)

	model = model.to(device)
	cudnn.benchmark = True

	optimizer = torch.optim.SGD(model.parameters(), 0.1,
								momentum=0.9, nesterov=True,
								weight_decay=5e-4)

	# cosine learning rate
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs)

	best_validation_loss = np.inf
	patience=0
	warmup_iters = 20*(len(train_loader))
	print("warmup_iters ", warmup_iters)
	iters = 0
	lrate = 0.1
	for epoch in range(0, epochs):
		# train for one epoch
		iters, train_loss = train_reject(iters, warmup_iters, lrate, train_loader, model, optimizer, scheduler, epoch, expert_fn, n_dataset, alpha)
		metrics = metrics_print(model, expert_fn, n_dataset, test_loader)
		print(metrics)
		validation_loss = metrics["validation_loss"]
		wandb.log({"training_loss": train_loss, "validation_loss": validation_loss})
		if validation_loss < best_validation_loss:
			best_validation_loss = validation_loss
			print("Saving the model with classifier accuracy {}".format(metrics['classifier_accuracy']), flush=True)
			torch.save(model.state_dict(), save_path + '.pt')
			patience=0
		else:
			patience+=1
		if patience >= 50:
			print("Early Exiting Training.", flush=True)
			break

n_dataset = 7  # cifar-10

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


def set_seed(seed):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
	from data_utils import *
	import wandb
	train, val, test = ham10000_expert.read(data_aug=True)
	print(len(train), len(val))
	
	for seed in [ 948,  625,  436,  791, 1750,  812, 1331, 1617,  650, 1816]:
		for alpha in [0.0, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0, 2.5]:
			bsz=256
			n_dataset=7
			print("seed and alpha are ", seed, alpha, flush=True)
			config = {
				"setting": "OvA HAM10000 expert sampling alpha experiment",
				"epochs" : 200,
				"patience": 50,
				"batch_size": 1024,
				"initial_lr" : 0.1,
				"warmup_iters" : "20 epochs",
				"seed" : seed

			}
			run = wandb.init(project="OvA_ham10000_different_seed_alpha_experiment", config=config, reinit=True, entity="aritzz")
			path = './Models/' + 'OvA_expert_sampling_seed_' + str(seed) + '_alpha_' + str(alpha)
			expert = synth_expert()
			model = ResNet34_defer(n_dataset+1)
			run_reject(model, train, val, n_dataset, expert.predict, 200, alpha, bsz, save_path=path) # train for 200 epochs
			run.finish()