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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResNet34_triage(nn.Module):
	def __init__(self, out_size):
		super(ResNet34_defer, self).__init__()
		self.resnet34 = torchvision.models.resnet34(pretrained=True)
		num_ftrs = self.resnet34.fc.in_features
		self.resnet34.fc = nn.Sequential(
			nn.Linear(num_ftrs, out_size))
		self.log_softmax = nn.LogSoftmax(dim=-1)

	def forward(self, x):
		x = self.resnet34(x)
		return self.log_softmax(x)


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
		outs = torch.argmax(conf, dim=1) #conf.multinomial(num_samples=1, replacement=True)
		return outs.cpu().tolist()


def train_confidence(trainloader, validloader, expert_fn, constraint):
	machine_type = 'confidence'
	print('-----training machine model using constraint:',constraint,' and machine model: ',machine_type)
	data = load_data(data_path)
	X = torch.from_numpy(data['X']).float()
	Y = torch.from_numpy(data['Y']).long()
	hconf = torch.mean(data['hconf']) + torch.zeros(X.shape[0])
	
	val_X = torch.from_numpy(data['val']['X']).float()
	val_Y = torch.from_numpy(data['val']['Y']).long()
	val_hconf = torch.mean(data['hconf']) + torch.zeros(val_X.shape[0])
	
	batch_size = 128
	num_batches = int(X.shape[0] / batch_size)
	val_num_batches = int(val_X.shape[0] / batch_size)
	
	num_epochs = 30
		
	mnet = models.resnet50()
	mnet.fc = torch.nn.Sequential(
		nn.Linear(mnet.fc.in_features, 2),
		nn.LogSoftmax(dim = -1)
	)
	mnet.to(device)
		
	optimizer = torch.optim.Adam(mnet.parameters(),lr=0.004)
	loss_func = torch.nn.NLLLoss(reduction='none')
	train_losses = []
	val_losses = []
	best_val_loss = 1000
	max_patience = 10
	patience = 0
	eps = 1e-4
	for epoch in range(num_epochs):
		print('----- epoch:',epoch, '-----')
		train_loss = 0
		with torch.no_grad():
			mprim = copy.deepcopy(mnet)
		machine_loss = []
		for i in range(num_batches):
			X_batch = X[i * batch_size: (i + 1) * batch_size].to(device)
			Y_batch = Y[i * batch_size: (i + 1) * batch_size].to(device)
			hconf_batch = hconf[i * batch_size: (i + 1) * batch_size].to(device)
			machine_scores_batch = mprim(X_batch)
			machine_conf_batch, _ = torch.max(machine_scores_batch,axis = 1)  
			machine_indices = find_machine_samples(hconf_batch,machine_conf_batch,constraint)
				
			X_machine = X_batch[machine_indices]
			Y_machine = Y_batch[machine_indices]
			optimizer.zero_grad()
			loss = loss_func(mnet(X_machine),Y_machine)
			loss.sum().backward()
			optimizer.step()
			train_loss += float(loss.mean())

		train_losses.append(train_loss / num_batches)
		print('machine_loss:', train_loss/num_batches)
		
		with torch.no_grad():
			val_loss = 0
			for i in range(val_num_batches):
				val_X_batch = val_X[i * batch_size: (i + 1) * batch_size].to(device)
				val_Y_batch = val_Y[i * batch_size: (i + 1) * batch_size].to(device)
				val_hconf_batch = val_hconf[i * batch_size: (i + 1) * batch_size].to(device)
				val_machine_scores = mprim(val_X_batch)
				val_machine_conf,_ = torch.max(val_machine_scores,axis=1)
				val_machine_indices = find_machine_samples(val_hconf_batch,val_machine_conf,constraint)
				val_loss += float(loss_func(mnet(val_X_batch[val_machine_indices]),val_Y_batch[val_machine_indices]).mean())

				
			val_loss /= val_num_batches
			print('val_loss:',val_loss) 

			if val_loss + eps < best_val_loss:
				torch.save(mnet.state_dict(), model_dir + 'm_' + machine_type + str(constraint))
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
 

if __name__ == "__main__":
	train_data, val_data, _ = ham10000_expert.read(data_aug=True)
	batch_size = 512

	kwargs = {'num_workers': 0, 'pin_memory': True}
	trainloader = torch.utils.data.DataLoader(train_data,
										   batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
	validloader = torch.utils.data.DataLoader(val_data,
											batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

	expert = synth_expert()

	print("Confidence Baseline...")
	for confidence in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
		print("confidence ", confidence)
		train_confidence(trainloader, validloader, expert.predict, confidence)