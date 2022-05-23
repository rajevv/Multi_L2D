import torch.nn as nn
import torch
import numpy as np




class Criterion(object):
	def __init__(self):
		pass
		
	def softmax(self, outputs,labels, collection_Ms, n_classes):
		'''
		The L_{CE} loss implementation for CIFAR
		----
		outputs: network outputs
		m: cost of deferring to expert cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
		labels: target
		m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
		n_classes: number of classes
		'''
		batch_size = outputs.size()[0]  # batch_size
		m2 = collection_Ms[0][1]
		rcs = []
		for i, _ in enumerate(collection_Ms, 0):
			rcs.append([n_classes-(i+1)] * batch_size)
		
		temp = -m2 * torch.log2(outputs[range(batch_size), labels])
		for i, (m,_) in enumerate(collection_Ms):
			temp -= m * torch.log2(outputs[range(batch_size), rcs[len(rcs)-1-i]])  
		return torch.sum(temp) / batch_size

	# Still ToDo to be compatible with multiple experts
	def ova(self, outputs, m, labels, m2, n_classes):
		batch_size = outputs.size()[0]
		l1 = Criterion.LogisticLoss(outputs[range(batch_size), labels], 1)
		l2 = torch.sum(Criterion.LogisticLoss(outputs[:,:n_classes], -1), dim=1) - Criterion.LogisticLoss(outputs[range(batch_size),labels],-1)
		l3 = Criterion.LogisticLoss(outputs[range(batch_size), n_classes], -1)
		l4 = Criterion.LogisticLoss(outputs[range(batch_size), n_classes], 1)

		l5 = m * (l4 - l3)

		l = m2 * (l1 + l2) + l3 + l5

		return torch.mean(l)

	@staticmethod
	def LogisticLoss(outputs, y):
		outputs[torch.where(outputs==0.0)] = (-1*y)*(-1*np.inf)
		l = torch.log2(1 + torch.exp((-1*y)*outputs))
		return l