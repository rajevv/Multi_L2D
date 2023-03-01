# To include lib
import sys

sys.path.insert(0, '../')

import argparse
import json
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
from ham10000dataset import ham10000_expert
from models.expert_model import MLPMixer
from models.experts import synth_expert
from models.resnet34 import ResNet34_defer
from torch.autograd import Variable

from lib.losses import Criterion
from lib.utils import AverageMeter, accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


def set_seed(seed):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)


def evaluate(model,
			 expert_fns,
			 loss_fn,
			 n_classes,
			 data_loader,
			 config):
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
	#  === Individual Expert Accuracies === #
	expert_correct_dic = {k: 0 for k in range(len(expert_fns))}
	expert_total_dic = {k: 0 for k in range(len(expert_fns))}
	#  === Individual  Expert Accuracies === #
	alpha = config["alpha"]
	losses = []
	with torch.no_grad():
		for data in data_loader:
			images, labels, Z = data
			images, labels, Z = images.to(device), labels.to(device), Z.float().to(device)
			outputs = model(images)
			if config["loss_type"] == "softmax":
				outputs = F.softmax(outputs, dim=1)
			if config["loss_type"] == "ova":
				ouputs = F.sigmoid(outputs)

			_, predicted = torch.max(outputs.data, 1)
			batch_size = outputs.size()[0]  # batch_size

			expert_predictions = []
			collection_Ms = []  # a collection of 3-tuple
			for i, fn in enumerate(expert_fns, 0):
				exp_prediction1 = fn(images, labels, Z)
				m = [0] * batch_size
				m2 = [0] * batch_size
				for j in range(0, batch_size):
					if exp_prediction1[j] == labels[j].item():
						m[j] = 1
						m2[j] = alpha
					else:
						m[j] = 0
						m2[j] = 1

				m = torch.tensor(m)
				m2 = torch.tensor(m2)
				m = m.to(device)
				m2 = m2.to(device)
				collection_Ms.append((m, m2))
				expert_predictions.append(exp_prediction1)

			loss = loss_fn(outputs, labels, collection_Ms, n_classes)
			losses.append(loss.item())

			for i in range(0, batch_size):
				r = (predicted[i].item() >= n_classes - len(expert_fns))
				prediction = predicted[i]
				if predicted[i] >= n_classes - len(expert_fns):
					max_idx = 0
					# get second max
					for j in range(0, n_classes - len(expert_fns)):
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
					# deferred_exp = (predicted[i] - (n_classes - len(expert_fns))).item()
					deferred_exp = ((n_classes - 1) - predicted[i]).item()  # reverse order, as in loss function
					exp_prediction = expert_predictions[deferred_exp][i]
					#
					# Deferral accuracy: No matter expert ===
					exp += (exp_prediction == labels[i].item())
					exp_total += 1
					# Individual Expert Accuracy ===
					expert_correct_dic[deferred_exp] += (exp_prediction == labels[i].item())
					expert_total_dic[deferred_exp] += 1
					#
					correct_sys += (exp_prediction == labels[i].item())
				real_total += 1
	cov = str(total) + str(" out of") + str(real_total)

	#  === Individual Expert Accuracies === #
	expert_accuracies = {"expert_{}".format(str(k)): 100 * expert_correct_dic[k] / (expert_total_dic[k] + 0.0002) for k
						 in range(len(expert_fns))}
	# Add expert accuracies dict
	to_print = {"coverage": cov, "system_accuracy": 100 * correct_sys / real_total,
				"expert_accuracy": 100 * exp / (exp_total + 0.0002),
				"classifier_accuracy": 100 * correct / (total + 0.0001),
				"alone_classifier": 100 * alone_correct / real_total,
				"validation_loss": np.average(losses),
				"n_experts": len(expert_fns),
				**expert_accuracies}
	print(to_print, flush=True)
	return to_print



def train_epoch(iters,
				warmup_iters,
				lrate,
				train_loader,
				model,
				optimizer,
				scheduler,
				epoch,
				expert_fns,
				loss_fn,
				n_classes,
				alpha,
				config):
	""" Train for one epoch """

	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	model.train()
	end = time.time()

	epoch_train_loss = []

	for i, (input, target, Z) in enumerate(train_loader):
		if iters < warmup_iters:
			lr = lrate * float(iters) / warmup_iters
			# print(iters, lr)
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr

		target = target.to(device)
		input = input.to(device)
		Z = Z.float().to(device)

		# compute output
		output = model(input)

		if config["loss_type"] == "softmax":
			output = F.softmax(output, dim=1)

		# get expert  predictions and costs
		batch_size = output.size()[0]  # batch_size
		collection_Ms = []
		# We only support \alpha=1
		for _, fn in enumerate(expert_fns):
			# We assume each expert function has access to the extra metadata, even if they don't use it.
			m = fn(input, target, Z)
			m2 = [0] * batch_size
			for j in range(0, batch_size):
				if m[j] == target[j].item():
					m[j] = 1
					m2[j] = alpha
				else:
					m[j] = 0
					m2[j] = 1
			m = torch.tensor(m)
			m2 = torch.tensor(m2)
			m = m.to(device)
			m2 = m2.to(device)
			collection_Ms.append((m, m2))

		# compute loss
		loss = loss_fn(output, target, collection_Ms, n_classes)
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
		iters += 1

		if i % 10 == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
				epoch, i, len(train_loader), batch_time=batch_time,
				loss=losses, top1=top1), flush=True)

	return iters, np.average(epoch_train_loss)



def train(model,
		  train_dataset,
		  validation_dataset,
		  expert_fns,
		  config):
	n_classes = config["n_classes"] + len(expert_fns)
	kwargs = {'num_workers': 0, 'pin_memory': True}
	train_loader = torch.utils.data.DataLoader(train_dataset,
											   batch_size=config["batch_size"], shuffle=True, drop_last=True, **kwargs)
	valid_loader = torch.utils.data.DataLoader(validation_dataset,
											   batch_size=config["batch_size"], shuffle=True, drop_last=True, **kwargs)
	model = model.to(device)
	cudnn.benchmark = True
	optimizer = torch.optim.SGD(model.parameters(), config["lr"],
								momentum=0.9, nesterov=True,
								weight_decay=config["weight_decay"])
	criterion = Criterion()
	loss_fn = getattr(criterion, config["loss_type"])
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * config["epochs"])
	best_validation_loss = np.inf
	patience = 0
	iters = 0
	warmup_iters = config["warmup_epochs"] * len(train_loader)
	lrate = config["lr"]

	for epoch in range(0, config["epochs"]):
		iters, train_loss = train_epoch(iters,
										warmup_iters,
										lrate,
										train_loader,
										model,
										optimizer,
										scheduler,
										epoch,
										expert_fns,
										loss_fn,
										n_classes,
										config["alpha"],
										config)
		metrics = evaluate(model,
						   expert_fns,
						   loss_fn,
						   n_classes,
						   valid_loader,
						   config)

		validation_loss = metrics["validation_loss"]

		if validation_loss < best_validation_loss:
			best_validation_loss = validation_loss
			print("Saving the model with classifier accuracy {}".format(metrics['classifier_accuracy']), flush=True)
			save_path = os.path.join(config["ckp_dir"],
									 config["experiment_name"] + '_' + str(len(expert_fns)) + '_experts')
			torch.save(model.state_dict(), save_path + '.pt')
			# Additionally save the whole config dict
			with open(save_path + '.json', "w") as f:
				json.dump(config, f)
			patience = 0
		else:
			patience += 1

		if patience >= config["patience"]:
			print("Early Exiting Training.", flush=True)
			break


# === Experiment 1 === #
def increase_experts(config):
	config["ckp_dir"] = "./" + config["loss_type"] + "_increase_experts"
	os.makedirs(config["ckp_dir"], exist_ok=True)

	experiment_experts = [1, 2, 4, 6, 8, 12, 14, 16, 18, 20]
	# experiment_experts = [1, 2]
	# experiment_experts = [4, 6]
	# experiment_experts = [8]

	# experiment_experts = [config["n_experts"]]
	for n in experiment_experts:
		print(n)
		num_experts = n
		expert = synth_expert(device=device)
		expert_fn = getattr(expert, config["expert_type"])
		expert_fns = [expert_fn] * n
		model = model = ResNet34_defer(int(config["n_classes"])+num_experts) #WideResNet(28, 3, int(config["n_classes"]) + num_experts, 4, dropRate=0.0)
		trainD, valD, _ = ham10000_expert.read(data_aug=True)
		train(model, trainD, valD, expert_fns, config)





if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--batch_size", type=int, default=1024)
	parser.add_argument("--alpha", type=float, default=1.0,
						help="scaling parameter for the loss function, default=1.0.")
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--patience", type=int, default=20,
						help="number of patience steps for early stopping the training.")
	parser.add_argument("--expert_type", type=str, default="MLPMixer",
						help="specify the expert type. For the type of experts available, see-> models -> experts. defualt=predict.")
	parser.add_argument("--n_classes", type=int, default=7,
						help="K for K class classification.")
	parser.add_argument("--k", type=int, default=5)
	# Dani experiments =====
	parser.add_argument("--n_experts", type=int, default=2)
	# Dani experiments =====
	parser.add_argument("--lr", type=float, default=0.1,
						help="learning rate.")
	parser.add_argument("--weight_decay", type=float, default=5e-4)
	parser.add_argument("--warmup_epochs", type=int, default=20)
	parser.add_argument("--loss_type", type=str, default="softmax",
						help="surrogate loss type for learning to defer.")
	parser.add_argument("--ckp_dir", type=str, default="./Models",
						help="directory name to save the checkpoints.")
	parser.add_argument("--experiment_name", type=str, default="multiple_experts",
						help="specify the experiment name. Checkpoints will be saved with this name.")

	config = parser.parse_args().__dict__

	print(config)
	increase_experts(config)