import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import pickle5 as pickle
from data_utils import *
from expert_model import MLPMixer
import script_ours_copy
import script_ovaloss
import confidence
import score_baseline
import baselines
import torchvision.models as models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def load_data(file_name):
	assert(os.path.exists(file_name+'.pkl'))
	with open(file_name + '.pkl', 'rb') as f:
		data = pickle.load(f)
	return data



def save_data(data, file_path):
	with open(file_path + '.pkl','wb') as f:
		pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)


# #### The Galaxy-zoo data can be found [here](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge).  Data is prepared in "prepare_data.py" and saved in "galaxy_data.pkl". Refer to section 6 of the paper for a detailed description of preprocessing and human predictions modeling. In this notebook we only load the data which we previously generated using the "prepare_data.py".


constraints = [0.0,0.2,0.4,0.6,0.8,1.0]
data_path = 'galaxy_data'
model_dir = './Models/'
res_dir = './Results/'
if not os.path.exists(model_dir):
	os.mkdir(model_dir)
if not os.path.exists(res_dir):
	os.mkdir(res_dir)




def get_test_assignments_us(seed, n_dataset, testloader, constraints, model_dir='./Models/'):
	machine_type = 'Differentiable'
	if seed != '':
			mc = machine_type + '_seed_' + str(seed) + '_'
	else:
			mc = machine_type + '_'

	res_path = res_dir + mc
	res = {}

	threshold_map = {0.0:0.0,0.2:0.257,0.4:0.257,0.6:0.211,0.8:0.211,1.0:0.373}
	
	with torch.no_grad():
		loss_func = torch.nn.NLLLoss(reduction='none')

		expert = baselines.synth_expert()
		expert_fn = expert.predict
		
		
		for constraint in constraints:
			print("constraint ", constraint)
			mnet = baselines.ResNet34_triage(n_dataset)
			mnet.to(device)

			mnet.load_state_dict(torch.load(model_dir + 'm_' + mc + str(constraint)))
			mnet.eval()

			gnet = baselines.ResNet34_triage(2)
			gnet.to(device)

			gnet.load_state_dict(torch.load(model_dir + 'g_' + mc + str(constraint)))
			gnet.eval()

			res[constraint] = {}
			l = []
			for i, batch in enumerate(testloader):
				test_x, test_y, test_z = batch
				test_x = test_x.to(device)
				test_y = test_y.to(device)
				test_z = test_z.float().to(device)
				_, hlabel_b = expert_fn(test_x, test_y, test_z)
				loss = np.zeros(test_x.shape[0])
				num_machine = int((1.0 - constraint) * test_x.shape[0])

				mscores = mnet(test_x)
				machine_conf, _ = torch.max(mscores,axis = 1)
				mlabel = torch.argmax(mscores,dim=1)

				gprediction = torch.exp(gnet(test_x).detach()[:,1])
				human_candidates = torch.argsort(gprediction)[num_machine:]
				to_machine = [i for i in range(mlabel.shape[0]) if i not in human_candidates or gprediction[i]<threshold_map[constraint]]
			
				to_human = np.array([i for i in range(mlabel.shape[0]) if i not in to_machine])

				print('number of samples to machine: ',len(to_machine),'number of samples to human:', len(to_human))
			
				mloss = np.not_equal(mlabel.cpu().data.numpy(),test_y.cpu().data.numpy())
				hloss = np.not_equal(hlabel_b.cpu().data.numpy(),test_y.cpu().data.numpy())
			
				if len(to_machine)!=0:
					loss[to_machine] = mloss[to_machine]
					print('mean of machine error:' ,np.mean(loss[to_machine]))
				if to_human.shape[0]!=0:
					loss[to_human] = hloss[to_human]
					print('mean of human error:' ,np.mean(loss[to_human]))

				l.append(np.mean(loss))
			res[constraint]['agg_loss'] = np.mean(l)
		save_data(res,res_path)


def get_test_assignments_surrogate(seed, constraints, model, expert_fn, testloader, model_dir, alpha):
	with torch.no_grad():
		machine_type = 'sontag_expert_sampling'
		# res_path = res_dir + machine_type
		res = {}
		
		model.to(device)

		if seed != '':
			mc = machine_type + '_seed_' + str(seed) + '_alpha_' + str(alpha)
		else:
			mc = machine_type + '_alpha_' + str(alpha)

		res_path = res_dir + mc + '_alpha_' + str(alpha)

		model.load_state_dict(torch.load(model_dir + mc + '.pt', map_location=device))
		model.eval()

		losses = []
		num_batches = len(testloader)

		for constraint in constraints:
			res[constraint] = {}
			l = []
			for i, batch in enumerate(testloader):
				X_batch, Y_batch, Z_batch = batch
				X_batch = X_batch.to(device)
				Y_batch = Y_batch.data.numpy()
				Z_batch = Z_batch.float().to(device)
				mscores, _ = model(X_batch)


				mlabel = torch.argmax(mscores[:,:-1],dim=1).cpu().data.numpy()
				hlabel_b = expert_fn(X_batch, Y_batch, Z_batch)
				hlabel_b = np.array(hlabel_b)

		
				last_class_prob = mscores[:,-1]
				highest_prob,_ = torch.max(mscores[:,:-1],dim=1)
				diff = last_class_prob - highest_prob

				
				to_defer = torch.where(diff > 0)[0]

				ids,_ = torch.sort(to_defer)

				loss = np.zeros(X_batch.shape[0])
				num_machine = int((1.0-constraint) * X_batch.shape[0])
				human = torch.argsort(diff)[num_machine:].cpu().data.numpy()

				print("length ", constraint, len(ids), len(human))

				if len(human) >= len(ids):
					human_candidates = ids
				else:
					human_candidates = human

				to_machine = [i for i in range(mlabel.shape[0]) if i not in human_candidates]
			
				to_human = np.array([i for i in range(mlabel.shape[0]) if i not in to_machine])

				mloss = np.not_equal(mlabel,Y_batch)
				hloss = np.not_equal(hlabel_b,Y_batch)

				if len(to_machine)!=0:
					loss[to_machine] = mloss[to_machine]
					print('mean of machine error:' ,np.mean(loss[to_machine]))
				if to_human.shape[0]!=0:
					loss[to_human] = hloss[to_human]
					print('mean of human error:' ,np.mean(loss[to_human]))
				
				l.append(np.mean(loss))

			res[constraint]['agg_loss'] = np.mean(l)
		save_data(res,res_path)


def get_test_assignments_ova(seed, constraints, model, expert_fn, testloader, model_dir, alpha):
	# loss_func = torch.nn.NLLLoss(reduction='none')
	with torch.no_grad():
		machine_type = 'OvA_expert_sampling'
		
		res = {}
		
		model.to(device)

		if seed != '':
			mc = machine_type + '_seed_' + str(seed) + '_alpha_' + str(alpha)
		else:
			mc = machine_type + '_alpha_' + str(alpha)

		res_path = res_dir + mc + '_alpha_' + str(alpha)

		model.load_state_dict(torch.load(model_dir + mc + '.pt', map_location=device))
		model.eval()

		losses = []
		num_batches = len(testloader)

		for constraint in constraints:
			res[constraint] = {}
			l = []
			for i, batch in enumerate(testloader):
				X_batch, Y_batch, Z_batch = batch
				X_batch = X_batch.to(device)
				Y_batch = Y_batch.data.numpy()
				Z_batch = Z_batch.float().to(device)

				_, mscores = model(X_batch)
				mlabel = torch.argmax(mscores[:,:-1],dim=1).cpu().data.numpy()

				hlabel_b = expert_fn(X_batch, Y_batch, Z_batch)
				hlabel_b = np.array(hlabel_b)
		
				last_class_prob = mscores[:,-1]
				highest_prob,_ = torch.max(mscores[:,:-1],dim=1)
				diff = last_class_prob - highest_prob

				
				to_defer = torch.where(diff > 0)[0]

				ids,_ = torch.sort(to_defer)

				loss = np.zeros(X_batch.shape[0])
				num_machine = int((1.0-constraint) * X_batch.shape[0])
				human = torch.argsort(diff)[num_machine:].cpu().data.numpy()

				print("length ", constraint, len(ids), len(human))

				if len(human) >= len(ids):
					human_candidates = ids
				else:
					human_candidates = human

				to_machine = [i for i in range(mlabel.shape[0]) if i not in human_candidates]
			
				to_human = np.array([i for i in range(mlabel.shape[0]) if i not in to_machine])

				mloss = np.not_equal(mlabel,Y_batch)
				hloss = np.not_equal(hlabel_b,Y_batch)

				if len(to_machine)!=0:
					loss[to_machine] = mloss[to_machine]
					print('mean of machine error:' ,np.mean(loss[to_machine]))
				if to_human.shape[0]!=0:
					loss[to_human] = hloss[to_human]
					print('mean of human error:' ,np.mean(loss[to_human]))
				
				l.append(np.mean(loss))

			res[constraint]['agg_loss'] = np.mean(l)
		save_data(res,res_path)



def get_test_assignments_confidence(seed, n_dataset, testloader, constraints, model_dir='./Models/'):
	machine_type = 'confidence'
	if seed != '':
		mc = machine_type + '_seed_' + str(seed) + '_'
	else:
		mc = machine_type + '_'
	res_path= res_dir + mc
	res = {}
	with torch.no_grad():
		loss_func = torch.nn.NLLLoss(reduction='none')
		expert = confidence.synth_expert()
		expert_fn = expert.predict
		hconf = []
		for i, batch in enumerate(testloader):
			X, Y, Z = batch
			X = X.to(device)
			Y = Y.to(device)
			Z = Z.float().to(device)
			conf,_ = expert_fn(X, Y, Z)
			hconf.extend(conf)
		test_hconf = torch.from_numpy(np.array(hconf))
		test_hconf = torch.mean(test_hconf)

		for constraint in constraints:
			res[constraint] = {}
			l = []
			mnet = confidence.ResNet34_confidence(n_dataset)
			mnet.to(device)

			mnet.load_state_dict(torch.load(model_dir + 'm_' + mc + str(constraint)))
			mnet.eval()

			for i, batch in enumerate(testloader):
				test_X_batch, test_Y_batch, test_Z_batch = batch
				test_X_batch = test_X_batch.to(device)
				test_Y_batch = test_Y_batch.to(device)
				test_Z_batch = test_Z_batch.float().to(device)

				_, hlabel = expert_fn(test_X_batch, test_Y_batch, test_Z_batch)
				hlabel = hlabel.cpu().data.numpy()

				loss = np.zeros(test_X_batch.shape[0])
		  
				machine_scores = mnet(test_X_batch)
				mcrossloss = loss_func(machine_scores,test_Y_batch.to(device))
				mlabel = torch.argmax(machine_scores,dim=1).cpu().data.numpy()
				machine_conf, _ = torch.max(machine_scores,axis = 1)  
				num_machine = int((1.0-constraint) * test_X_batch.shape[0])
				hconf = test_hconf + torch.zeros(test_X_batch.shape[0])
				hconf = hconf.to(device)
				to_machine = confidence.find_machine_samples(hconf,machine_conf,constraint).cpu().data.numpy()

				to_human = np.array([i for i in range(test_X_batch.shape[0]) if i not in to_machine])

				mloss = np.not_equal(mlabel,test_Y_batch.cpu().data.numpy())
				hloss = np.not_equal(hlabel,test_Y_batch.cpu().data.numpy())

				if len(to_machine)!=0:
					loss[to_machine] = mloss[to_machine]
					print('mean of machine error:' ,np.mean(loss[to_machine]))
				if to_human.shape[0]!=0:
					loss[to_human] = hloss[to_human]
					print('mean of human error:' ,np.mean(loss[to_human]))
				l.append(np.mean(loss))
			res[constraint]['agg_loss'] = np.mean(l)
		save_data(res,res_path)


def get_assignments_score(seed, n_dataset, testloader, constraints, model_dir='./Models/'):
	with torch.no_grad():
		machine_type = 'score'
		if seed != '':
			mc = machine_type + '_seed_' + str(seed)
		else:
			mc = machine_type
		res_path = res_dir + mc

		res = {}
		loss_func = torch.nn.NLLLoss(reduction='none')
		expert = score_baseline.synth_expert()
		expert_fn = expert.predict

		mnet = score_baseline.ResNet34_triage(n_dataset)

		mnet.load_state_dict(torch.load(model_dir + 'm_' + mc))
		mnet.to(device)
		mnet.eval()
		losses = []
		for constraint in constraints:
			res[constraint] = {}
			l = []
			for i, batch in enumerate(testloader):
				test_X_batch, test_Y_batch, test_Z_batch = batch
				test_X_batch = test_X_batch.to(device)
				test_Y_batch = test_Y_batch.to(device)
				test_Z_batch = test_Z_batch.float().to(device)
				hlabel = expert_fn(test_X_batch, test_Y_batch, test_Z_batch)
				hlabel = hlabel.cpu().data.numpy()
				mscores = mnet(test_X_batch)
				mcrossloss = loss_func(mscores, test_Y_batch)
				mlabel = torch.argmax(mscores,dim=1).cpu().data.numpy()
				mconf,_ = torch.max(mscores,axis = 1)


				loss = np.zeros(test_X_batch.shape[0])
				num_machine = int((1.0-constraint) * test_X_batch.shape[0])
				to_machine = torch.argsort(mconf,descending = True)[:num_machine].cpu().data.numpy()
				to_human = np.array([i for i in range(test_X_batch.shape[0]) if i not in to_machine])

				mloss = np.not_equal(mlabel,test_Y_batch.cpu().data.numpy())
				hloss = np.not_equal(hlabel,test_Y_batch.cpu().data.numpy())
			
				if len(to_machine)!=0:
					loss[to_machine] = mloss[to_machine]
					print('mean of machine error:' ,np.mean(loss[to_machine]))
				if to_human.shape[0]!=0:
					loss[to_human] = hloss[to_human]
					print('mean of human error:' ,np.mean(loss[to_human]))
				l.append(np.mean(loss))
			res[constraint]['agg_loss'] = np.mean(l)
		save_data(res,res_path)


if __name__ == "__main__":
	# get_test_assignments_us(constraints)

	batch_size = 256
	_,_, test_data = ham10000_expert.read(data_aug=False)
	kwargs = {'num_workers': 0, 'pin_memory': True}
	testloader = torch.utils.data.DataLoader(test_data,
											   batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
	# print("surrogate...")
	n_dataset = 7

	print("surrogate")
	for seed in [948,  625]:#,  436,  791, 1750,  812, 1331, 1617,  650, 1816]:
		for alpha in [0.0, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0, 2.5]:
			print('seed and alpha are ', seed, alpha, flush=True)
			model_sontag = script_ours_copy.ResNet34_defer(n_dataset+1)
			expert_sontag = script_ours_copy.synth_expert()

			get_test_assignments_surrogate(seed, constraints, model_sontag, expert_sontag.predict, testloader, './Models/', alpha)
	
	print("surrogate_ova")

	for seed in [948,  625]: #,  436,  791, 1750,  812, 1331, 1617,  650, 1816]:
		for alpha in [0.0, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0, 2.5]:
			print('seed: ', seed, flush=True)
			model_ova = script_ovaloss.ResNet34_defer(n_dataset+1)
			expert_ova = script_ovaloss.synth_expert()
			get_test_assignments_ova(seed, constraints, model_ova, expert_ova.predict, testloader, './Models/', alpha)

	# print("confidence")
	# for seed in ['', 948, 625, 436, 791, 1750]:
	# 	print('seed: ', seed, flush=True)
	# 	get_test_assignments_confidence(seed, n_dataset, testloader, constraints, model_dir='./Models/')

	# print("score")
	# for seed in ['', 948, 625, 436, 791, 1750]:
	# 	print('seed: ', seed, flush=True)
	# 	get_assignments_score(seed, n_dataset, testloader, constraints, model_dir='./Models/')

	# print("differentiable")
	# for seed in ['', 948, 625, 436, 791, 1750]:
	# 	print('seed: ', seed, flush=True)
	# 	get_test_assignments_us(seed, n_dataset, testloader, constraints, model_dir='./Models/')

