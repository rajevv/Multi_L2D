# To include lib
import sys

sys.path.insert(0, '../')

import argparse
import json
import os
from collections import defaultdict

from cifar10dataset import cifar
from main_gradual_overlap import evaluate
from models.experts import synth_expert2
from models.wideresnet import WideResNet

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

def forward(model, dataloader, expert_fns, n_classes, n_experts):
	confidence = []
	true = []
	expert_predictions = defaultdict(list)
	# density = []

	with torch.no_grad():
		for inp, lbl in dataloader:
			inp = inp.to(device)
			conf = model(inp)
			for i, fn in enumerate(expert_fns):
				expert_pred1 = fn(inp, lbl)
				expert_predictions[i].append(expert_pred1)
			confidence.append(conf.cpu())
			true.append(lbl[:, 0])


	true = torch.stack(true, dim=0).view(-1)
	confidence = torch.stack(confidence, dim=0).view(-1, n_classes + n_experts)
	for k, v in expert_predictions.items():
		expert_predictions[k] = torch.stack([torch.tensor(k) for k in v], dim=0).view(-1)

	print(true.shape, confidence.shape, [v.shape for k, v in
										 expert_predictions.items()])  # ,expert_predictions1.shape, expert_predictions2.shape) #, density.shape)
	return true, confidence, [v.numpy() for k, v in
							  expert_predictions.items()]  # (expert_predictions1, expert_predictions2) #, density


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)


def validation(model_name, expert_fns, config, seed=""):
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
		true_label[severity] = true.numpy()
		classifier_confidence[severity] = confidence.numpy()
		expert_preds[severity] = expert_predictions
		result[severity] = filter(result_)

	result = {}
	classifier_confidence = {}
	true_label = {}
	expert_preds = {}

	n_dataset = 10
	batch_size = 1024
	n_expert = len(expert_fns)

	# Data ===
	ood_d, test_d = cifar.read(severity=0, slice_=-1, test=True, only_id=True)

	kwargs = {'num_workers': 1, 'pin_memory': True}
	test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

	# Model ===
	model = WideResNet(28, 3, n_dataset + num_experts, 4, dropRate=0.0)
	model_path = os.path.join(config["ckp_dir"], config["experiment_name"] + '_' + str(
		len(expert_fns)) + '_experts' +  '.pt') #'_seed_' + str(seed) +
	model.load_state_dict(torch.load(model_path, map_location=device))
	model = model.to(device)

	get('test', test_dl)

	with open(config["ckp_dir"] + 'true_label_multiple_experts_new' + model_name + '.txt', 'w') as f:
		json.dump(json.dumps(true_label, cls=NumpyEncoder), f)

	with open(config["ckp_dir"] + 'confidence_multiple_experts_new' + model_name + '.txt', 'w') as f:
		json.dump(json.dumps(classifier_confidence, cls=NumpyEncoder), f)

	with open(config["ckp_dir"] + 'expert_predictions_multiple_experts_new' + model_name + '.txt', 'w') as f:
		json.dump(json.dumps(expert_preds, cls=NumpyEncoder), f)

	with open(config["ckp_dir"] + 'validation_multiple_experts_new' + model_name + '.txt', 'w') as f:
		json.dump(json.dumps(result, cls=NumpyEncoder), f)
	return result


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--batch_size", type=int, default=1024)
	parser.add_argument("--alpha", type=float, default=1.0,
						help="scaling parameter for the loss function, default=1.0.")
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--patience", type=int, default=20,
						help="number of patience steps for early stopping the training.")
	parser.add_argument("--expert_type", type=str, default="predict_prob_cifar_2",
						help="specify the expert type. For the type of experts available, see-> models -> experts. defualt=predict.")
	parser.add_argument("--n_classes", type=int, default=10,
						help="K for K class classification.")
	parser.add_argument("--k", type=int, default=5)
	# Dani experiments =====
	parser.add_argument("--n_experts", type=int, default=2)
	# Dani experiments =====
	parser.add_argument("--lr", type=float, default=0.1,
						help="learning rate.")
	parser.add_argument("--weight_decay", type=float, default=5e-4)
	parser.add_argument("--warmup_epochs", type=int, default=0)
	parser.add_argument("--loss_type", type=str, default="softmax",
						help="surrogate loss type for learning to defer.")
	parser.add_argument("--ckp_dir", type=str, default="./Models",
						help="directory name to save the checkpoints.")
	parser.add_argument("--experiment_name", type=str, default="multiple_experts",
						help="specify the experiment name. Checkpoints will be saved with this name.")

	config = parser.parse_args().__dict__
	config["ckp_dir"] = "./" + config["loss_type"] + "_increase_experts_nonoverlapping/"
	print(config)

	alpha = 1.0
	n_dataset = 10
	seeds = [""] #948, 625, 436]
	accuracy = []

	for seed in seeds:
		print("seed is {}".format(seed))
		if seed != "":
			set_seed(seed)
		acc = []
		expert_fns = []
		for i, n in enumerate([1,2,3,4,5,6,7,8,9,10]): #[1, 4, 8, 12, 16, 20]):
			print("n is {}".format(n))
			model_name = '_' + str(n) + '_experts' #+ '_seed_' + str(seed)
			num_experts = n

			# Expert ===
			expert = synth_expert2(S=[i], p_in=1.0, p_out=0.0, n_classes=config["n_classes"])
			expert_fn = getattr(expert, config["expert_type"])
			expert_fns.append(expert_fn)


			result = validation(model_name, expert_fns, config, seed=seed)
			acc.append(result["test"]["system_accuracy"])
		accuracy.append(acc)

	print("===Mean and Standard Error===")
	print(np.mean(np.array(accuracy), axis=0))
	print(stats.sem(np.array(accuracy), axis=0))

