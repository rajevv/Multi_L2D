import json

import numpy as np
from scipy import stats
import torch
#import matplotlib.pyplot as plt
from reliability_diagram import compute_calibration

# global variable
random_expert_index = 0
num_experts = 4

def get_softmax_ECEs(seed=""):
	# === Softmax ===
	n_classes = 10
	# confs = []
	# exps = []
	# true = []
	path = "softmax_increase_confidence/"
	p_experts = [0.2, 0.4, 0.6, 0.8, 0.95]
	ECE = []
	ECE_random = []
	for p_in in p_experts:
		model_name = '_' + str(p_in) + '_confidence' + '_seed_' + str(seed) # 
		with open(path + 'confidence_multiple_experts' + model_name + '.txt', 'r') as f:
			conf = json.loads(json.load(f))
		with open(path + 'expert_predictions_multiple_experts' + model_name + '.txt', 'r') as f:
			exp_pred = json.loads(json.load(f))
		with open(path + 'true_label_multiple_experts' + model_name + '.txt', 'r') as f:
			true_label = json.loads(json.load(f))
		c = torch.tensor(conf['test'])
		#print(type(true_label['test']), type(exp_pred['test']), type(c))
		#print(np.array(true_label['test']).shape, np.array(exp_pred['test']).shape, c.shape)
		c = c.softmax(dim=1)
		#c = c.sigmoid()

		temp = 0
		for i in range(num_experts):
			temp += c[:, n_classes + i]
		prob = c / (1.0 - temp).unsqueeze(-1)
		# print(prob.shape)
		# confs.append(prob)

		true_label = torch.tensor(true_label['test'])
		exp_prediction = torch.tensor(exp_pred['test']) # exp_prediction shape [n, test_data_size]

		# ECEs ===

		# enumerate over all the experts
		eces = []
		for j in range(num_experts):
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
		ECE_random.append(eces[random_expert_index])
	return ECE, ECE_random

#print("Softmax Increasing # Experts ECE: {}".format(ECE))


def get_OvA_ECEs(seed=""):
	# === OvA ===

	n_classes = 10
	path = "ova_increase_confidence/"
	p_experts = [0.2, 0.4, 0.6, 0.8, 0.95]
	ECE = []
	ECE_random = []
	for p_in in p_experts:
		model_name = '_' + str(p_in) + '_confidence' + '_seed_' + str(seed) 
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
		for j in range(num_experts):
			e_j = exp_prediction[j,:]
			t = true_label
			c_j = prob[:, n_classes + j]
			acc_j = t.eq(e_j)
			log = compute_calibration(c_j, acc_j)
			eces.append(log['expected_calibration_error'].numpy())
		ECE.append(np.average(eces))
		ECE_random.append(eces[random_expert_index])
	return ECE, ECE_random

# if __name__ == "__main__":
print("---In main---")
seeds = [948, 625, 436]
ECE_softmax = []
ECE_OvA = []

for seed in seeds:
	ECE_softmax.append(get_softmax_ECEs(seed=seed)[0])
	ECE_OvA.append(get_OvA_ECEs(seed=seed)[0])

print("---AVERAGE---")

print("===Mean and Standard Error ECEs Softmax===")
print("All \n {}".format(np.array(ECE_softmax)))
print("Mean {}".format(np.mean(np.array(ECE_softmax), axis=0)))
print("Standard Error {}".format(stats.sem(np.array(ECE_softmax), axis=0)))

print("===Mean and Standard Error ECEs OvA===")
print("All \n {}".format(np.array(ECE_OvA)))
print("Mean {}".format(np.mean(np.array(ECE_OvA), axis=0)))
print("Standard Error {}".format(stats.sem(np.array(ECE_OvA), axis=0)))

print("---RANDOM EXPERT---")

ECE_softmax = []
ECE_OvA = []

for seed in seeds:
	ECE_softmax.append(get_softmax_ECEs(seed=seed)[1])
	ECE_OvA.append(get_OvA_ECEs(seed=seed)[1])

print(np.array(ECE_softmax).squeeze(axis=2).shape)

print("===Mean and Standard Error Random Expert ECE Softmax===")
print("All \n {}".format(np.array(ECE_softmax).squeeze(axis=2)))
print("Mean {}".format(np.mean(np.array(ECE_softmax).squeeze(axis=2), axis=0)))
print("Standard Error {}".format(stats.sem(np.array(ECE_softmax).squeeze(axis=2), axis=0)))

print("===Mean and Standard Error Random Expert ECE OvA===")
print("All \n {}".format(np.array(ECE_OvA).squeeze(axis=2)))
print("Mean {}".format(np.mean(np.array(ECE_OvA).squeeze(axis=2), axis=0)))
print("Standard Error {}".format(stats.sem(np.array(ECE_OvA).squeeze(axis=2), axis=0)))





# def get_softmax_ECEs
# # === Softmax ===
# n_classes = 10
# random_expert_idx = 0
# confs = []
# exps = []
# true = []
# path = "softmax_increase_confidence/"
# n_experts = 4
# p_experts = [0.2, 0.4, 0.6, 0.8, 0.95]
# for p in p_experts:
#     model_name = '_' + str(p) + '_confidence'
#     with open(path + 'confidence_multiple_experts' + model_name + '.txt', 'r') as f:
#         conf = json.loads(json.load(f))
#     with open(path + 'expert_predictions_multiple_experts' + model_name + '.txt', 'r') as f:
#         exp_pred = json.loads(json.load(f))
#     with open(path + 'true_label_multiple_experts' + model_name + '.txt', 'r') as f:
#         true_label = json.loads(json.load(f))
#     true.append(true_label['test'])
#     exps.append(exp_pred['test'])
#     c = torch.tensor(conf['test'])
#     print(c.shape)
#     # DANI Correction ===
#     c = c.softmax(dim=1)
#     # DANI Correction ===

#     temp = 0
#     for i in range(n_experts):
#         temp += c[:, (n_classes + n_experts) - (i + 1)]
#     prob = c / (1.0 - temp).unsqueeze(-1)
#     confs.append(prob)

# ECEs = []
# for i in range(len(p_experts)):
#     c = confs[i]
#     e = exps[i]
#     t = torch.tensor(true[i])
#     eces = []

#     for j in range(n_experts):
#         e_j = e[j]
#         c_j = c[:, c.shape[1] - (j + 1)]
#         t_j = t
#         ids_where_gt_one = torch.where(c_j > 1.0)
#         c_j[ids_where_gt_one] = 1.0
#         acc_j = t_j.eq(torch.tensor(e_j))
#         log = compute_calibration(c_j, acc_j)
#         eces.append(log['expected_calibration_error'].numpy())
#     ECEs.append(eces)

# Y = []
# # average on all experts
# for l in ECEs:
#     Y.append(np.average(l))

# print("Softmax Average among Experts ECE: {}".format(Y))
# plt.plot(Y)
# plt.show()

# Y_random = []
# # random expert ECE
# for l in ECEs:
#     Y_random.append(l[random_expert_idx][0])

# print("Softmax Random Expert ECE: {}".format(Y_random))
# plt.plot(Y_random)
# plt.show()

# # === OvA ===
# n_classes = 10
# random_expert_idx = 0
# confs = []
# exps = []
# true = []
# path = "ova_increase_confidence/"
# n_experts = 4
# p_experts = [0.2, 0.4, 0.6, 0.8, 0.95]
# for p in p_experts:
#     model_name = '_' + str(p) + '_confidence'
#     with open(path + 'confidence_multiple_experts' + model_name + '.txt', 'r') as f:
#         conf = json.loads(json.load(f))
#     with open(path + 'expert_predictions_multiple_experts' + model_name + '.txt', 'r') as f:
#         exp_pred = json.loads(json.load(f))
#     with open(path + 'true_label_multiple_experts' + model_name + '.txt', 'r') as f:
#         true_label = json.loads(json.load(f))
#     true.append(true_label['test'])
#     exps.append(exp_pred['test'])
#     c = torch.tensor(conf['test'])
#     # DANI Correction ===
#     c = c.sigmoid()
#     # DANI Correction ===
#     confs.append(c)

# ECEs = []
# for i in range(len(p_experts)):
#     c = confs[i]
#     e = exps[i]
#     t = torch.tensor(true[i])
#     eces = []

#     for j in range(n_experts):
#         e_j = e[j]
#         c_j = c[:, c.shape[1] - (j + 1)]
#         t_j = t
#         ids_where_gt_one = torch.where(c_j > 1.0)
#         c_j[ids_where_gt_one] = 1.0
#         acc_j = t_j.eq(torch.tensor(e_j))
#         log = compute_calibration(c_j, acc_j)
#         eces.append(log['expected_calibration_error'].numpy())
#     ECEs.append(eces)

# Y = []
# # average on all experts
# for l in ECEs:
#     Y.append(np.average(l))

# print("OvA Average among Experts ECE: {}".format(Y))
# plt.plot(Y)
# plt.show()

# Y_random = []
# # random expert ECE
# for l in ECEs:
#     Y_random.append(l[random_expert_idx][0])

# print("OvA Random Expert ECE: {}".format(Y_random))
# plt.plot(Y_random)
# plt.show()


