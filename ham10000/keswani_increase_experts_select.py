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
import json
from utils import *
from data_utils import *
from models.keswani_models import *
from models.experts import *
from losses.losses import *

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

	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	classifier, deferrer = model[0], model[1]
	classifier.eval()
	deferrer.eval()

	end = time.time()

	epoch_train_loss = []

	for i, (input, target, Z) in enumerate(data_loader):

		target = target.to(device)
		input = input.to(device)
		Z = Z.float().to(device)

		# compute classifier prediction
		clf_output = classifier(input)
		# classifier predictions
		_, clf_pred = torch.max(clf_output.data, 1)


		# get expert  predictions and costs
		batch_size = clf_pred.size()[0]
		collection_Ms = []

		for _, fn in enumerate(expert_fns):
			# We assume each expert function has access to the extra metadata, even if they don't use it.
			m = fn(input, target, Z)
			exp_pred = torch.zeros((batch_size, config["n_classes"]))
			for j in range(batch_size):
				exp_pred[j][m[j]] = 1
			collection_Ms.append(exp_pred)

		# Append classifier prediction to collection_Ms as well
		temp = torch.zeros((batch_size, config["n_classes"]))
		for j in range(batch_size):
			temp[j][clf_pred.cpu().tolist()[j]] = 1

		collection_Ms.append(temp)
		collection = torch.stack(collection_Ms).transpose(0,1) 
		collection = collection.transpose(1,2)
		collection = collection.to(device)

		# deferrer output
		def_output = deferrer(input).unsqueeze(-1) #[bsz, E+1, -1]

		# aggregation
		output = torch.bmm(collection, def_output).squeeze(-1)

		loss = 0.5*loss_fn(clf_output, target) + 0.5*loss_fn(output, target)

		# measure accuracy and record loss
		prec1 = accuracy(output.data, target, topk=(1,))[0]
		losses.update(loss.data.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))


	# print('Epoch: [{0}][{1}/{2}]\t'
	# 	  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
	# 	  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
	# 	  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
	# 	epoch, i, len(train_loader), batch_time=batch_time,
	# 	loss=losses, top1=top1), flush=True)

	# Add expert accuracies dict
	to_print = {"system_accuracy": top1.avg,
				"validation_loss": losses.avg,
				}
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

	# model.train()
	classifier, deferrer = model[0], model[1]
	classifier.train()
	deferrer.train()

	end = time.time()

	epoch_train_loss = []

	for i, (input, target, Z) in enumerate(train_loader):
		if iters < warmup_iters:
			lr = lrate * float(iters) / warmup_iters
			print(iters, lr)
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr

		target = target.to(device)
		input = input.to(device)
		Z = Z.float().to(device)

		# compute classifier prediction
		clf_output = classifier(input)
		# classifier predictions
		_, clf_pred = torch.max(clf_output.data, 1)

		#print("clf_pred shape {}".format(clf_pred.shape))

		# if config["loss_type"] == "softmax":
		# 	output = F.softmax(output, dim=1)

		# get expert  predictions and costs
		batch_size = clf_pred.size()[0]  # batch_size
		collection_Ms = []
		# We only support \alpha=1
		for _, fn in enumerate(expert_fns):
			# We assume each expert function has access to the extra metadata, even if they don't use it.
			m = fn(input, target, Z)
			exp_pred = torch.zeros((batch_size, config["n_classes"]))
			for j in range(batch_size):
				exp_pred[j][m[j]] = 1
			collection_Ms.append(exp_pred)

		# Append classifier prediction to collection_Ms as well
		temp = torch.zeros((batch_size, config["n_classes"]))
		for j in range(batch_size):
			temp[j][clf_pred.cpu().tolist()[j]] = 1

		collection_Ms.append(temp)
		collection = torch.stack(collection_Ms).transpose(0,1) #torch.tensor(collection_Ms).T  #[bsz, E+1]
		collection = collection.transpose(1,2)
		collection = collection.to(device)

		# deferrer output
		def_output = deferrer(input).unsqueeze(-1) #[bsz, E+1, -1]

		# aggregation
		output = torch.bmm(collection, def_output).squeeze(-1)

	

		# compute loss
		loss = 0.5*loss_fn(clf_output, target) + 0.5*loss_fn(output, target)
		epoch_train_loss.append(loss.item())

		# measure accuracy and record loss
		prec1 = accuracy(output.data, target, topk=(1,))[0]
		losses.update(loss.data.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))

		# # compute gradient and do SGD step
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
		  config,
		  seed=""):
	n_classes = config["n_classes"] + len(expert_fns)
	kwargs = {'num_workers': 0, 'pin_memory': True}
	train_loader = torch.utils.data.DataLoader(train_dataset,
											   batch_size=config["batch_size"], shuffle=True, drop_last=True, **kwargs)
	valid_loader = torch.utils.data.DataLoader(validation_dataset,
											   batch_size=config["batch_size"], shuffle=True, drop_last=True, **kwargs)
	model = (model[0].to(device), model[1].to(device))
	cudnn.benchmark = True
	optimizer = torch.optim.SGD(list(model[0].parameters()) + list(model[1].parameters()), config["lr"],
								momentum=0.9, nesterov=True,
								weight_decay=config["weight_decay"])
	criterion = nn.CrossEntropyLoss()
	loss_fn = criterion #getattr(criterion, config["loss_type"])
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
		# break
		metrics = evaluate(model,
						   expert_fns,
						   loss_fn,
						   n_classes,
						   valid_loader,
						   config)

		# validation_loss = metrics["validation_loss"]

		# if validation_loss < best_validation_loss:
		# 	best_validation_loss = validation_loss
		# 	print("Saving the model with classifier accuracy {}".format(metrics['classifier_accuracy']), flush=True)
		# 	save_path = os.path.join(config["ckp_dir"],
		# 							 config["experiment_name"] + '_' + str(len(expert_fns)) + '_experts' + '_seed_' + str(seed))
		# 	torch.save(model.state_dict(), save_path + '.pt')
		# 	# Additionally save the whole config dict
		# 	with open(save_path + '.json', "w") as f:
		# 		json.dump(config, f)
		# 	patience = 0
		# else:
		# 	patience += 1

		# if patience >= config["patience"]:
		# 	print("Early Exiting Training.", flush=True)
		# 	break


# === Experiment 1 === #
all_available_experts = ['MLPMixer', 'predict', 'predict_prob', 'predict_random']
def increase_experts(config):
	config["ckp_dir"] = "./" + config["loss_type"] + "_keswani_increase_experts_select"
	os.makedirs(config["ckp_dir"], exist_ok=True)

	experiment_experts = [4, 6, 12, 16, 2, 1, 8]
	# experiment_experts = [1, 2]
	# experiment_experts = [4, 6]
	# experiment_experts = [8]

	# experiment_experts = [config["n_experts"]]
	for seed in [948, 625, 436]:
		print("run for seed {}".format(seed))
		set_seed(seed)
		log = {'selected_experts' : {}}
		for n in experiment_experts:
			print("n is {}".format(n))
			num_experts = n
			selected_experts = random.choices(all_available_experts,k=n)
			print("selected experts {}".format(selected_experts))
			log['selected_experts'][n] = selected_experts
			expert = synth_expert()
			expert_fns = []
			for expert_type in selected_experts:
				expert_fn = getattr(expert, expert_type)
				expert_fns.append(expert_fn)
			classifier = Classifier(out_size = int(config["n_classes"]))
			deferrer = Deferrer(out_size = num_experts + 1)
			model = (classifier, deferrer)
			trainD, valD, _ = ham10000_expert.read(data_aug=True)
			train(model, trainD, valD, expert_fns, config, seed=seed)
			break
		break
		pth = os.path.join(config['ckp_dir'], config['experiment_name'] + '_log_' + '_seed_' + str(seed))
		with open(pth + '.json', 'w') as f:
			json.dump(log, f)




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--batch_size", type=int, default=1024)
	parser.add_argument("--alpha", type=float, default=1.0,
						help="scaling parameter for the loss function, default=1.0.")
	parser.add_argument("--epochs", type=int, default=150)
	parser.add_argument("--patience", type=int, default=50,
						help="number of patience steps for early stopping the training.")
	parser.add_argument("--expert_type", type=str, default="MLPMixer",
						help="specify the expert type. For the type of experts available, see-> models -> experts. defualt=predict.")
	parser.add_argument("--n_classes", type=int, default=7,
						help="K for K class classification.")
	parser.add_argument("--k", type=int, default=5)
	# Dani experiments =====
	parser.add_argument("--n_experts", type=int, default=2)
	# Dani experiments =====
	parser.add_argument("--lr", type=float, default=0.001,
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