import math
import torch
import torch.nn as nn
import random
import numpy as np
from scipy import stats
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
from collections import defaultdict
from expert_model import MLPMixer
import json
from utils import *
from data_utils import *
from models.resnet34 import *
from models.experts import *
from losses.losses import *
from main_increase_experts_select import *

def set_seed(seed):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)

def forward(model, dataloader, expert_fns, n_classes, n_experts):
	confidence = []
	true = []
	expert_predictions = defaultdict(list)

	with torch.no_grad():
		for inp, lbl, Z in dataloader:
			inp = inp.to(device)
			lbl = lbl.to(device)
			Z = Z.float().to(device)
			conf = model(inp)
			for i, fn in enumerate(expert_fns):
				expert_pred1 = fn(inp, lbl, Z)
				expert_predictions[i].append(expert_pred1)
			confidence.append(conf.cpu())
			true.append(lbl.cpu())
			#break

	true = torch.stack(true, dim=0).view(-1)
	confidence = torch.stack(confidence, dim=0).view(-1, n_classes + n_experts)
	for k, v in expert_predictions.items():
		expert_predictions[k] = torch.stack([torch.tensor(k) for k in v], dim=0).view(-1)

	#print(true.shape, confidence.shape, [v.shape for k, v in
										 #expert_predictions.items()])  
	return true, confidence, [v.numpy() for k, v in
							  expert_predictions.items()] 

class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)


def validation(model_name, expert_fns, config):
	def filter(dict_):
		d = {}
		for k, v in dict_.items():
			if torch.is_tensor(v):
				v = v.item()
			d[k] = v
		return d

	def get(severity, dl):
		true, confidence, expert_predictions = forward(model, dl, expert_fns, n_dataset, n_expert)

		print("shapes: true labels {}, confidences {}, expert_predictions {}".format(\
			true.shape, confidence.shape, np.array(expert_predictions).shape))

		criterion = Criterion()
		loss_fn = getattr(criterion, config["loss_type"])
		n_classes = n_dataset
		print("Evaluate...")
		result_ = evaluate(model, expert_fns, loss_fn, n_classes+len(expert_fns), dl, config)
		#print(result_)
		result[severity]  = result_
		true_label[severity] = true.numpy()
		classifier_confidence[severity] = confidence.numpy()
		expert_preds[severity] = expert_predictions

	result = {}
	classifier_confidence = {}
	true_label = {}
	expert_preds = {}

	n_dataset = config["n_classes"]
	batch_size = config["batch_size"]
	num_experts = len(expert_fns)
	n_expert = num_experts
	
	# Data ===
	_,_, test_d = ham10000_expert.read(data_aug=False)

	kwargs = {'num_workers': 1, 'pin_memory': True}
	test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

	# Model ===
	model = ResNet34_defer(int(config["n_classes"])+num_experts)
	model_path = os.path.join(config["ckp_dir"], config["experiment_name"] + '_' + model_name + '.pt')
	model.load_state_dict(torch.load(model_path, map_location=device))
	model = model.to(device)

	get('test', test_dl)

	with open(config["ckp_dir"] + 'true_label_multiple_experts_' + model_name + '.txt', 'w') as f:
		json.dump(json.dumps(true_label, cls=NumpyEncoder), f)

	with open(config["ckp_dir"] + 'confidence_multiple_experts_' + model_name + '.txt', 'w') as f:
		json.dump(json.dumps(classifier_confidence, cls=NumpyEncoder), f)

	with open(config["ckp_dir"] + 'expert_predictions_multiple_experts_' + model_name + '.txt', 'w') as f:
		json.dump(json.dumps(expert_preds, cls=NumpyEncoder), f)

	with open(config["ckp_dir"] + 'validation_results_' + model_name + '.txt', 'w') as f:
		json.dump(json.dumps(result, cls=NumpyEncoder), f)

	return result


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--alpha", type=float, default=1.0,
						help="scaling parameter for the loss function, default=1.0.")

	parser.add_argument("--n_classes", type=int, default=7,
						help="K for K class classification.")
	parser.add_argument("--k", type=int, default=5)

	parser.add_argument("--loss_type", type=str, default="softmax",
						help="surrogate loss type for learning to defer.")
	parser.add_argument("--ckp_dir", type=str, default="./Models",
						help="directory name to save the checkpoints.")
	parser.add_argument("--experiment_name", type=str, default="multiple_experts",
						help="specify the experiment name. Checkpoints will be saved with this name.")

	config = parser.parse_args().__dict__

	config["ckp_dir"] = './' + config["loss_type"] + '_increase_experts_select/'

	experiment_experts = [1, 2, 4, 6, 8, 12, 16]
	seeds = [948, 625, 436]

	accuracy = []


	for seed in seeds:
		print("run for seed {}".format(seed))
		set_seed(seed)
		acc = []
		json_path = os.path.join(config['ckp_dir'], config['experiment_name'] + '_log_' + '_seed_' + str(seed))
		with open(json_path + '.json', 'r') as f:
			log = json.load(f)
		for n in experiment_experts:
			print("n is {}".format(n))
			num_experts = n
			selected_experts = log["selected_experts"][str(n)]
			print("selected experts {}".format(selected_experts))
			expert = synth_expert()
			expert_fns = []
			for expert_type in selected_experts:
				expert_fn = getattr(expert, expert_type)
				expert_fns.append(expert_fn)

			model_name = str(len(expert_fns)) + '_experts' + '_seed_' + str(seed)
			result = validation(model_name, expert_fns, config)
			acc.append(result['test']['system_accuracy'])

		accuracy.append(acc)

	print("===Mean and Standard Error===")
	print("Mean {}".format(np.mean(np.array(accuracy), axis=0)))
	print("Standard Error {}".format(stats.sem(np.array(accuracy), axis=0)))




		# 	break
		# break