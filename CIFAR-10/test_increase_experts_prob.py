import json

import numpy as np
from scipy import stats
import torch
#import matplotlib.pyplot as plt
from reliability_diagram import compute_calibration

def get_softmax_ECEs(seed=""):
	# === Softmax ===
	n_classes = 10
	# confs = []
	# exps = []
	# true = []
	path = "softmax_increase_experts_prob/"
	n_experts = [1, 4, 8, 12, 16, 20]
	ECE = []
	for n in n_experts:
		model_name = '_' + 'new' + '_' + str(n) + '_experts' + '_seed_' + str(seed) # 
		with open(path + 'confidence_multiple_experts' + model_name + '.txt', 'r') as f:
			conf = json.loads(json.load(f))
		with open(path + 'expert_predictions_multiple_experts' + model_name + '.txt', 'r') as f:
			exp_pred = json.loads(json.load(f))
		with open(path + 'true_label_multiple_experts' + model_name + '.txt', 'r') as f:
			true_label = json.loads(json.load(f))
		# true.append(true_label['test'])
		# exps.append(exp_pred['test'])
		c = torch.tensor(conf['test'])
		#print(type(true_label['test']), type(exp_pred['test']), type(c))
		#print(np.array(true_label['test']).shape, np.array(exp_pred['test']).shape, c.shape)
		c = c.softmax(dim=1)
		#c = c.sigmoid()

		temp = 0
		for i in range(n):
			temp += c[:, n_classes + i]
		prob = c / (1.0 - temp).unsqueeze(-1)
		# print(prob.shape)
		# confs.append(prob)

		true_label = torch.tensor(true_label['test'])
		exp_prediction = torch.tensor(exp_pred['test']) # exp_prediction shape [n, test_data_size]

		# ECEs ===

		# enumerate over all the experts
		eces = []
		for j in range(n):
			e_j = exp_prediction[j,:]
			t = true_label
			c_j = prob[:, n_classes + j]
			ids_where_gt_one = torch.where(c_j > 1.0)
			c_j[ids_where_gt_one] = 1.0
			acc_j = t.eq(e_j)
			log = compute_calibration(c_j, acc_j)
			eces.append(log['expected_calibration_error'].numpy())
		#print(eces)
		ECE.append(np.average(eces))
	return ECE

#print("Softmax Increasing # Experts ECE: {}".format(ECE))


def get_OvA_ECEs(seed=""):
	# === OvA ===

	n_classes = 10
	path = "ova_increase_experts_prob/"
	n_experts = [1, 4, 8, 12, 16, 20]
	ECE = []
	for n in n_experts:
		model_name = '_' + 'new' + '_' + str(n) + '_experts' + '_seed_' + str(seed) 
		with open(path + 'confidence_multiple_experts' + model_name + '.txt', 'r') as f:
			conf = json.loads(json.load(f))
		with open(path + 'expert_predictions_multiple_experts' + model_name + '.txt', 'r') as f:
			exp_pred = json.loads(json.load(f))
		with open(path + 'true_label_multiple_experts' + model_name + '.txt', 'r') as f:
			true_label = json.loads(json.load(f))
		# true.append(true_label['test'])
		# exps.append(exp_pred['test'])
		c = torch.tensor(conf['test'])
		#print(type(true_label['test']), type(exp_pred['test']), type(c))
		#print(np.array(true_label['test']).shape, np.array(exp_pred['test']).shape, c.shape)
		c = c.sigmoid()

		prob = c

		true_label = torch.tensor(true_label['test'])
		exp_prediction = torch.tensor(exp_pred['test']) # exp_prediction shape [n, test_data_size]

		# ECEs ===

		# enumerate over all the experts
		eces = []
		for j in range(n):
			e_j = exp_prediction[j,:]
			t = true_label
			c_j = prob[:, n_classes + j]
			acc_j = t.eq(e_j)
			log = compute_calibration(c_j, acc_j)
			eces.append(log['expected_calibration_error'].numpy())
		ECE.append(np.average(eces))
	return ECE

# if __name__ == "__main__":
print("---In main---")
seeds = [948]#, 625, 436]
ECE_softmax = []
ECE_OvA = []

for seed in seeds:
	ECE_softmax.append(get_softmax_ECEs(seed=seed))
	ECE_OvA.append(get_OvA_ECEs(seed=seed))

print("===Mean and Standard Error ECEs Softmax===")
print("All \n {}".format(np.array(ECE_softmax)))
print("Mean {}".format(np.mean(np.array(ECE_softmax), axis=0)))
print("Standard Error {}".format(stats.sem(np.array(ECE_softmax), axis=0)))

print("===Mean and Standard Error ECEs OvA===")
print("All \n {}".format(np.array(ECE_OvA)))
print("Mean {}".format(np.mean(np.array(ECE_OvA), axis=0)))
print("Standard Error {}".format(stats.sem(np.array(ECE_OvA), axis=0)))

#print("OvA Increasing # Experts ECE: {}".format(ECE))


