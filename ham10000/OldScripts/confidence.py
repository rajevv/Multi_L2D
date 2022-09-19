import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import os
import pickle5 as pickle
import torchvision.models as models
import torchvision
from data_utils import *
from expert_model import MLPMixer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(file_name):
	assert(os.path.exists(file_name+'.pkl'))
	with open(file_name + '.pkl', 'rb') as f:
		data = pickle.load(f)
	return data



def save_data(data, file_path):
	with open(file_path + '.pkl','wb') as f:
		pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)

def find_machine_samples(machine_loss, hloss, constraint):
	
	diff = machine_loss - hloss
	argsorted_diff = torch.clone(torch.argsort(diff))
	num_outsource = int(constraint * machine_loss.shape[0])
	index = -num_outsource

	while (index < -1 and diff[argsorted_diff[index]] <= 0):
		index += 1
	
	if index==0:
		index = -1
	if index == -diff.shape[0]:
		index = 1
	machine_list = argsorted_diff[:index]

	return machine_list




class ResNet34_confidence(nn.Module):
	def __init__(self, out_size):
		super(ResNet34_confidence, self).__init__()
		self.resnet34 = torchvision.models.resnet34(pretrained=True)
		num_ftrs = self.resnet34.fc.in_features
		self.resnet34.fc = nn.Sequential(
			nn.Linear(num_ftrs, out_size))
		self.log_softmax = nn.LogSoftmax(dim=-1)

	def forward(self, x):
		x = self.resnet34(x)
		return self.log_softmax(x)


def train_confidence(seed, n_dataset, trainloader, validloader, expert_fn, constraint, model_dir='./Models/'):
	machine_type = 'confidence'

	if seed != '':
		mct = machine_type + '_seed_' + str(seed) + '_'
	else:
		mct = machine_type + '_'

	print('-----training machine model using constraint:',constraint,' and machine model: ', machine_type)
	train_hconf = []
	for i, batch in enumerate(trainloader):
		X, Y, Z = batch
		X = X.to(device)
		Z = Z.float().to(device)
		hconf,_ = expert_fn(X, Y, Z)
		train_hconf.extend(hconf)
	train_hconf = torch.from_numpy(np.array(train_hconf))
	train_hconf = torch.mean(train_hconf)

	valid_hconf = []
	for i, batch in enumerate(validloader):
		X, Y, Z = batch
		X = X.to(device)
		Z = Z.float().to(device)
		hconf,_ = expert_fn(X, Y, Z)
		valid_hconf.extend(hconf)
	valid_hconf = torch.from_numpy(np.array(valid_hconf))
	valid_hconf = torch.mean(valid_hconf)
	
	num_epochs = 200
		
	mnet = ResNet34_confidence(n_dataset)
	mnet.to(device)
		
	optimizer = torch.optim.Adam(mnet.parameters(),lr=0.004)
	loss_func = torch.nn.NLLLoss(reduction='none')
	train_losses = []
	val_losses = []
	best_val_loss = 1000
	max_patience = 50
	patience = 0
	eps = 1e-4
	for epoch in range(num_epochs):
		print('----- epoch:',epoch, '-----')
		train_loss = 0
		with torch.no_grad():
			mprim = copy.deepcopy(mnet)
		machine_loss = []
		for i, batch in enumerate(trainloader):
			X_batch, Y_batch, _ = batch
			X_batch = X_batch.to(device)
			Y_batch = Y_batch.to(device)
			hconf_batch = train_hconf + torch.zeros(X_batch.shape[0])
			hconf_batch = hconf_batch.to(device)
			machine_scores_batch = mprim(X_batch)
			machine_conf_batch, _ = torch.max(machine_scores_batch,axis = 1)  
			machine_indices = find_machine_samples(hconf_batch,machine_conf_batch,constraint)
				
			X_machine = X_batch[machine_indices]
			Y_machine = Y_batch[machine_indices]
			optimizer.zero_grad()
			loss = loss_func(mnet(X_machine),Y_machine)
			loss.sum().backward()
			optimizer.step()
			train_loss += float(loss.cpu().mean())

		epoch_loss = train_loss / len(trainloader)
		
		with torch.no_grad():
			val_loss = 0
			for i, batch in enumerate(validloader):
				val_X_batch, val_Y_batch, _ =  batch
				val_X_batch = val_X_batch.to(device)
				val_Y_batch = val_Y_batch.to(device)

				val_hconf_batch = valid_hconf + torch.zeros(val_X_batch.shape[0])
				val_hconf_batch = val_hconf_batch.to(device)
				val_machine_scores = mprim(val_X_batch)
				val_machine_conf,_ = torch.max(val_machine_scores,axis=1)
				val_machine_indices = find_machine_samples(val_hconf_batch,val_machine_conf,constraint)
				val_loss += float(loss_func(mnet(val_X_batch[val_machine_indices]),val_Y_batch[val_machine_indices]).cpu().mean())

				
			val_loss /= len(validloader)

			wandb.log({"training_loss": epoch_loss, "validation_loss": val_loss})

			if val_loss + eps < best_val_loss:
				torch.save(mnet.state_dict(), model_dir + 'm_' + mct + str(constraint))
				best_val_loss = val_loss
				print('updated the model')
				patience = 0
			else:
				patience += 1
			val_losses.append(val_loss)

				
		if patience > max_patience:
			print('no progress for 10 epochs... stopping training')
			break
		print('\n')
		


# def get_test_assignments_confidence(constraints):
#     machine_type = 'confidence'
#     res_path= res_dir + machine_type
#     res = {}
#     with torch.no_grad():
#         loss_func = torch.nn.NLLLoss(reduction='none')
#         data = load_data(data_path)
#         test_X = torch.from_numpy(data['test']['X']).float().to(device)
#         test_Y = torch.from_numpy(data['test']['Y']).long()
#         hlabel = data['test']['hpred']
#         hconf = (torch.mean(data['hconf']) + torch.zeros(test_X.shape[0])).to(device)
#         hcrossloss = data['test']['hloss']

#         batch_size = 128
#         num_batches = int(test_X.shape[0]/batch_size)

#         losses = []
#         for constraint in constraints:
#             res[constraint] = {}
#             loss = np.zeros(test_X.shape[0])
#             mnet = models.resnet50()
#             mnet.fc = torch.nn.Sequential(
#             nn.Linear(mnet.fc.in_features, 2),
#             nn.LogSoftmax(dim = -1)
#             )
#             mnet.to(device)

#             mnet.load_state_dict(torch.load(model_dir + 'm_' + machine_type + str(constraint)))
#             mnet.eval()
#             loss_batches = 0
#             machine_scores = mnet(test_X)
#             mcrossloss = loss_func(machine_scores,test_Y.to(device))
#             mlabel = torch.argmax(machine_scores,dim=1).cpu().data.numpy()
#             machine_conf, _ = torch.max(machine_scores,axis = 1)  
#             num_machine = int((1.0-constraint) * test_X.shape[0])
#             to_machine = find_machine_samples(hconf,machine_conf,constraint).cpu().data.numpy()

#             to_human = np.array([i for i in range(test_X.shape[0]) if i not in to_machine])

#             mloss = np.not_equal(mlabel,test_Y)
#             hloss = np.not_equal(hlabel,test_Y)

#             loss[to_machine] = mloss[to_machine]
#             loss[to_human] =  hloss[to_human]
#             print(to_machine.shape[0],to_human.shape[0],test_X.shape[0])
#             print(np.mean(loss[to_machine]),np.mean(loss[to_human]))
#             print('-----')

#             losses.append(np.mean(loss))
            
#             res[constraint]['mpredloss'] = mloss
#             res[constraint]['mcrossloss'] = mcrossloss
#             res[constraint]['mlabel'] = mlabel
#             res[constraint]['mconf'] = machine_conf
#             res[constraint]['mscore'] = machine_scores
#             res[constraint]['hpredloss'] = hloss
#             res[constraint]['hcrossloss'] = hcrossloss
#             res[constraint]['hlabel'] = hlabel
#             res[constraint]['to_machine'] = to_machine
#             res[constraint]['to_human'] = to_human
#             res[constraint]['agg_loss'] = np.mean(loss)
        
#             del mnet

#         plt.plot(constraints,losses,marker='o')
#         plt.title(r'Confidence-based Triage',fontsize=22)
#         plt.ylabel(r'misclassification error',fontsize=22)
#         plt.xlabel(r'b',fontsize=22)
#         plt.show()



#         save_data(res,res_path)



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
		outs, pred = torch.max(conf, dim=1) #conf.multinomial(num_samples=1, replacement=True)
		return outs.cpu().tolist(), pred


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

	kwargs = {'num_workers': 0, 'pin_memory': True}

	batch_size=256
	trainloader = torch.utils.data.DataLoader(train,
											   batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
	validloader = torch.utils.data.DataLoader(val,
											   batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
	
	for seed in ['', 948,  625,  436,  791, 1750,  812, 1331, 1617,  650, 1816]:
		if seed != '':
			set_seed(seed)
		bsz=256
		n_dataset=7
		alpha=1.0
		print("seed ", seed, flush=True)
		expert = synth_expert()
		for confidence in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
			config = {
			"setting": "Confidence HAM10000 Expert Sampling",
			"epochs" : 200,
			"patience": 50,
			"batch_size": 256,
			"seed" : seed,
			"confidence" : confidence
			}
			run = wandb.init(project="confidence_ham10000_different_seeds", config=config, reinit=True, entity="aritzz")
			print("confidence ", confidence)
			train_confidence(seed, n_dataset, trainloader, validloader, expert.predict, confidence, model_dir='./Models/')
			run.finish()