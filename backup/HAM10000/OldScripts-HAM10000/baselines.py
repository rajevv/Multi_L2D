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




class ResNet34_triage(nn.Module):
	def __init__(self, out_size):
		super(ResNet34_triage, self).__init__()
		self.resnet34 = torchvision.models.resnet34(pretrained=True)
		num_ftrs = self.resnet34.fc.in_features
		self.resnet34.fc = nn.Sequential(
			nn.Linear(num_ftrs, out_size))
		self.log_softmax = nn.LogSoftmax(dim=-1)

	def forward(self, x):
		x = self.resnet34(x)
		return self.log_softmax(x)




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


def train_triage(seed, n_dataset, trainloader, validloader, expert_fn, constraint, model_dir='./Models/'):
	machine_type = 'Differentiable'
	if seed != '':
		mct = machine_type + '_seed_' + str(seed) + '_'
	else:
		mct = machine_type + '_'

	print('training machine model using constraint:',constraint,' and machine model: ',machine_type)
	
	
	num_epochs = 100
		
	mnet = ResNet34_triage(n_dataset)
	mnet.to(device)

	optimizer = torch.optim.Adam(mnet.parameters())
	loss_func = torch.nn.NLLLoss(reduction='none')
	train_losses = []
	val_losses = []
	labels = []
	best_val_loss = 1000
	eps = 1e-4
	max_patience = 20
	patience = 0
	res = {}
	res['machine_loss'] = {}
	
	
	for epoch in range(num_epochs):
		print('----- epoch:',epoch, '-----')
		train_loss = 0
		with torch.no_grad():
			mprim = copy.deepcopy(mnet)
		machine_loss = []
		for i, batch in enumerate(trainloader):
			X_batch, Y_batch, Z_batch = batch
			X_batch = X_batch.to(device)
			Y_batch = Y_batch.to(device)
			Z_batch = Z_batch.float().to(device)
			hloss_batch, _ = expert_fn(X_batch, Y_batch, Z_batch)
			with torch.no_grad():
				machine_scores_batch = mprim(X_batch)
				machine_loss_batch = loss_func(machine_scores_batch,Y_batch)
				machine_loss.extend(machine_loss_batch.detach())
				
			machine_indices = find_machine_samples(machine_loss_batch, hloss_batch, constraint)
			
			X_machine = X_batch[machine_indices]
			Y_machine = Y_batch[machine_indices]
			optimizer.zero_grad()
			loss = loss_func(mnet(X_machine),Y_machine)
			loss.sum().backward()
			optimizer.step()
			train_loss += float(loss.mean())

		epoch_loss = train_loss / len(trainloader)
		print('machine_loss:', epoch_loss)
		
		with torch.no_grad():
			val_loss = 0
			for i, batch in enumerate(validloader):
				val_X_batch, val_Y_batch, val_Z_batch = batch
				val_X_batch = val_X_batch.to(device)
				val_Y_batch = val_Y_batch.to(device)
				val_Z_batch = val_Z_batch.float().to(device)
				val_hloss_batch, _ = expert_fn(val_X_batch, val_Y_batch, val_Z_batch)
				val_mscores_batch = mprim(val_X_batch)
				val_mloss_batch = loss_func(val_mscores_batch, val_Y_batch)
				val_machine_indices = find_machine_samples(val_mloss_batch,val_hloss_batch,constraint)
				val_loss += float(loss_func(mnet(val_X_batch[val_machine_indices]),val_Y_batch[val_machine_indices]).mean())
				
				
			val_loss /= len(validloader)
			print('val_loss:',val_loss) 

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
		del mprim


def train_g(seed, n_dataset, trainloader, validloader, expert_fn, constraint, model_dir='./Models/'):
	machine_type = 'Differentiable'
	print('started training g using the constraint: ',constraint,' Using machine model: ',machine_type)
	if seed != '':
		mct = machine_type + '_seed_' + str(seed) + '_'

	else:
		mct = machine_type + '_'
	
	with torch.no_grad():
		mnet = ResNet34_triage(n_dataset)

		mnet.load_state_dict(torch.load(model_dir + 'm_' + mct + str(constraint)))
		mnet.to(device)
		mnet.eval()
	
	num_epochs = 100
	
	gnet = ResNet34_triage(2)
	gnet.to(device)


	g_optimizer = torch.optim.Adam(gnet.parameters(),lr=0.001)
	loss_func = torch.nn.NLLLoss(reduction='none')
	

	train_losses = []
	val_losses = []
	best_val_loss = 1000
	max_patience = 20
	patience = 0
	eps = 1e-4
	res = {}
	
	
	for epoch in range(num_epochs):
		machine_loss = []
		val_machine_loss = []
		gprediction = []
		val_gprediction = []
		glabels = []
		val_glabels = []
		print('----- epoch:',epoch, '-----')
		g_train_loss = 0
		for i, batch in enumerate(trainloader):
			batch_size = batch[0].shape[0]
			X_batch, Y_batch, Z_batch = batch
			X_batch = X_batch.to(device)
			Y_batch = Y_batch.to(device)
			Z_batch = Z_batch.float().to(device)
			hloss_batch, _ = expert_fn(X_batch, Y_batch, Z_batch)
			with torch.no_grad():
				machine_loss_batch = loss_func(mnet(X_batch),Y_batch)
				machine_indices = find_machine_samples(machine_loss_batch, hloss_batch, constraint)
			g_labels_batch = torch.tensor([0 if j in machine_indices else 1 for j in range(batch_size)]).to(device)
			g_optimizer.zero_grad()
			gpred = gnet(X_batch)
			g_loss = loss_func(gpred,g_labels_batch)
			g_loss.mean().backward()
			g_optimizer.step()
			g_train_loss += float(g_loss.mean())
			
		epoch_loss = g_train_loss/len(trainloader)
		print('g_loss:', epoch_loss) 
				  
		
		with torch.no_grad():
			val_gloss = 0
			for i, batch in enumerate(validloader):
				val_X_batch, val_Y_batch, val_Z_batch = batch
				val_X_batch = val_X_batch.to(device)
				val_Y_batch = val_Y_batch.to(device)
				val_Z_batch = val_Z_batch.float().to(device)
				val_hloss_batch, _ = expert_fn(val_X_batch, val_Y_batch, val_Z_batch)
				val_mscores = mnet(val_X_batch)
				val_machine_loss_batch = loss_func(val_mscores,val_Y_batch)
				val_machine_indices = find_machine_samples(val_machine_loss_batch,val_hloss_batch,constraint)
				val_glabels_batch = torch.tensor([0 if j in val_machine_indices else 1 for j in range(val_X_batch.shape[0])]).to(device)
				val_gpred = gnet(val_X_batch)
				val_loss = loss_func(val_gpred,val_glabels_batch)
				val_gloss += float(val_loss.mean())
				
			val_gloss /= len(validloader)
			print('val_g_loss:',float(val_gloss))

			wandb.log({"training_loss": epoch_loss, "validation_loss": val_loss})

			if val_gloss + eps < best_val_loss:
				torch.save(gnet.state_dict(), model_dir + 'g_' + mct + str(constraint))
				best_val_loss = val_gloss
				print('updated the model')
				patience = 0
			else:
				patience += 1

		if patience > max_patience:
			print('no progress for 20 epochs... stopping training')
			break
				
		print('\n')
		
	del gnet
	del mnet
	

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
		self.criterion = nn.NLLLoss(reduction='none')

	def predict(self, X, labels, Z):
		pred = self.expert(X, Z)
		# proxy for the expert's probability distribution
		conf = F.log_softmax(pred, dim=1)
		# sample from the expert's distribution
		pred = torch.argmax(conf, dim=1)
		l  = self.criterion(conf, labels) #conf.multinomial(num_samples=1, replacement=True)
		return l.detach(), pred





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

	expert = synth_expert()

	expert_fn = expert.predict
	
	# for seed in ['', 948,  625,  436,  791, 1750]: #  812, 1331, 1617,  650, 1816]:
	# 	if seed != '':
	# 			set_seed(seed)
	# 	for constraint in [0.0,0.2, 0.4, 0.6, 0.8, 1.0]:
	# 		n_dataset=7
	# 		alpha=1.0
	# 		print("seed ", seed, flush=True)
	# 		config = {
	# 		"setting": "Differentiable m model HAM10000 Expert Sampling",
	# 		"epochs" : 200,
	# 		"patience": 50,
	# 		"batch_size": 512,
	# 		"seed" : seed,
	# 		"confidence" : constraint
	# 		}
	# 		run = wandb.init(project="differentiable_m_model_ham10000_different_seeds", config=config, reinit=True, entity="aritzz")
	# 		train_triage(seed, n_dataset, trainloader, validloader, expert_fn, constraint, model_dir='./Models/')
	# 		run.finish()

	for seed in ['', 948,  625,  436,  791, 1750]: #,  812, 1331, 1617,  650, 1816]:
		if seed != '':
				set_seed(seed)
		for constraint in [0.0,0.2, 0.4, 0.6, 0.8, 1.0]:
			n_dataset=7
			alpha=1.0
			print("seed ", seed, flush=True)
			config = {
			"setting": "Differentiable g model HAM10000 Expert Sampling",
			"epochs" : 200,
			"patience": 50,
			"batch_size": 512,
			"seed" : seed,
			"confidence" : constraint
			}
			run = wandb.init(project="differentiable_g_model_ham10000_different_seeds", config=config, reinit=True, entity="aritzz")
			train_g(seed, n_dataset, trainloader, validloader, expert_fn, constraint, model_dir='./Models/')
			run.finish()