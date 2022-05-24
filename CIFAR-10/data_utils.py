from __future__ import division
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from scipy import io
import ddu_dirty_mnist
from sklearn.model_selection import train_test_split
import random



class cifar(Dataset):
	def __init__(self, data, labels, split_train_set=False, train_proportion=1.0, data_aug=False):
		super(cifar).__init__()

		self.data = data
		self.labels = labels
		#print("Dataset Size ", self.data.shape, self.labels.shape)

		if split_train_set:

			train_idx, _ = train_test_split(
											np.arange(self.labels.shape[0]),
											test_size = 1.0 - train_proportion,
											shuffle = True,
											stratify = self.labels[:,0].numpy()
											)
			
			self.data = self.data[train_idx]
			self.labels = self.labels[train_idx]
			print("Dataset Size after splitting ", self.data.shape, self.labels.shape)


		self.means = (self.data / 255.0).mean(axis=(0,2,3))#.unsqueeze(1).unsqueeze(2)
		self.stds = (self.data / 255.0).std(axis=(0,2,3))#.unsqueeze(1).unsqueeze(2)


		normalize = transforms.Normalize(mean = self.means,
				  std = self.stds)

		# self.severity = severity
		# normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
		# 		  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
		#self.normalize = transforms.Normalize(means,stds)
		if data_aug:
			self.transform = transforms.Compose([
				transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
												  (4, 4, 4, 4), mode='reflect').squeeze()),
				transforms.ToPILImage(),
				transforms.RandomCrop(32),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			])
		else:
				self.transform = transforms.Compose([
					normalize,
			])

	def normalize(self, image):
		return self.transform(image / 255.0)




	@classmethod
	def read(cls, severity=0, slice_=-1, test=False, data_aug=False, entropy=False, mix=False, only_id=False, only_ood=False,split_train_set=False, train_proportion=1.0):

		np.random.seed(0)
		random.seed(0)
		torch.manual_seed(0)

		if not test:
			if only_ood:
				(ood_train_data, ood_train_labels), (ood_val_data, ood_val_labels) = cifar.OOD(severity, slice_, test)
				return cls(ood_train_data, ood_train_labels, data_aug=data_aug), cls(ood_val_data, ood_val_labels, data_aug=False)

			if only_id:
				(id_train_data, id_train_labels), (id_val_data, id_val_labels) = cifar.ID(test)
				return cls(id_train_data, id_train_labels, data_aug=data_aug, split_train_set=split_train_set, train_proportion=train_proportion), cls(id_val_data, id_val_labels, data_aug=False)

			if entropy:
				(ood_train_data, ood_train_labels), (ood_val_data, ood_val_labels) = cifar.OOD(severity, slice_, test)
				(id_train_data, id_train_labels), (id_val_data, id_val_labels) = cifar.ID(test)
				return cls(id_val_data, id_val_labels, data_aug=False), cls(ood_val_data, ood_val_labels, data_aug=False)

			if mix:
				(ood_train_data, ood_train_labels), (ood_val_data, ood_val_labels) = cifar.OOD(severity, slice_, test)
				(id_train_data, id_train_labels), (id_val_data, id_val_labels) = cifar.ID(test)

				train_data = torch.cat((ood_train_data, id_train_data), 0)
				train_labels = torch.cat((ood_train_labels, id_train_labels), 0)

				val_data = torch.cat((ood_val_data, id_val_data), 0)
				val_labels = torch.cat((ood_val_labels, id_val_labels), 0)
				return cls(train_data, train_labels, data_aug=data_aug), cls(val_data, val_labels, data_aug=False)

		else:
			ood_data, ood_labels = cifar.OOD(severity, slice_, test)

			id_data, id_labels = cifar.ID(test)

			return cls(ood_data, ood_labels, data_aug=False), cls(id_data, id_labels, data_aug=False)

	@staticmethod
	def OOD(severity, slice_, test):
		corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost',\
				'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur',\
				'pixelate', 'shot_noise', 'snow', 'zoom_blur'][:slice_]
		
		path = './Data/CIFAR-10-C/'
		data = []
		labels = []
		lbl_file = np.load(path + 'labels.npy')
		for corruption in corruptions:
			#messy code but we are assuming we always train with the 0th perturbations
			data_ = np.load(path + corruption + '.npy')
			if severity != 0 and test == True:
				data.append(data_[10000*severity : (10000*severity+10000), :, :, :])
				labels.append(lbl_file[10000*severity : (10000*severity+10000)])
			elif severity == 0 and test == True:
				data.append(data_[8000:10000, :, :, :])
				labels.append(lbl_file[8000 : 10000])
			else:
				data.append(data_[10000*severity : (10000*severity+8000), :, :, :])
				labels.append(lbl_file[10000*severity : (10000*severity+8000)])
		


		
		data = torch.from_numpy(np.concatenate(data)).float().transpose(1,3)
		labels = torch.from_numpy(np.concatenate(labels)).unsqueeze(1)
		flag = torch.tensor([0]*labels.shape[0]).unsqueeze(1)
		labels = torch.cat((labels, flag), 1)

		if not test:
			idx = torch.randperm(data.shape[0])
			data = data[idx].view(data.size())
			labels = labels[idx].view(labels.size())

			train_size = int(0.90 * len(data))
			train_data = data[:train_size,:,:,:]
			train_labels = labels[:train_size,:]

			val_data = data[train_size:,:,:,:]
			val_labels = labels[train_size:,:]
			return (train_data, train_labels), (val_data, val_labels)
		
		else:

			return (data, labels)

	@staticmethod
	def ID(test):
		path = './Data/cifar/cifar-10-batches-py/'
		if not test:
			data = []
			labels = []
			for i in [1,2,3,4,5]:
				import pickle
				with open(path + 'data_batch_' + str(i), 'rb') as fo:
					dict = pickle.load(fo, encoding='bytes')
					#print(dict.keys())
					data.append(dict[b'data'])
					labels.append(dict[b'labels'])
			data = torch.from_numpy(np.concatenate(data)).float().reshape(50000, 3, 32, 32).transpose(2,3)
			labels = torch.from_numpy(np.concatenate(labels)).unsqueeze(1)
			flag = torch.tensor([1]*labels.shape[0]).unsqueeze(1)
			labels = torch.cat((labels, flag), 1)

			idx = torch.randperm(data.shape[0])
			data = data[idx].view(data.size())
			labels = labels[idx].view(labels.size())

			train_size = int(0.90 * len(data))
			train_data = data[:train_size,:,:,:]
			train_labels = labels[:train_size,:]

			val_data = data[train_size:,:,:,:]
			val_labels = labels[train_size:,:]
			return (train_data, train_labels), (val_data, val_labels)
		else:
			data = []
			labels = []
			import pickle
			with open(path + 'test_batch', 'rb') as fo:
				dict = pickle.load(fo, encoding='bytes')
				# print(dict.keys())
				data.append(dict[b'data'])
				labels.append(dict[b'labels'])
			data = torch.from_numpy(np.concatenate(data)).float().reshape(10000, 3, 32, 32).transpose(2,3)
			labels = torch.from_numpy(np.concatenate(labels)).unsqueeze(1)
			flag = torch.tensor([1]*labels.shape[0]).unsqueeze(1)
			labels = torch.cat((labels, flag), 1)
			return (data, labels)

	def __getitem__(self, index):
		return self.normalize(self.data[index]), self.labels[index]

	def __len__(self):
		return self.labels.shape[0]


class SVHN(Dataset):
	def __init__(self):
		mat = io.loadmat('./Data/svhn/test_32x32.mat')
		data = mat['X']
		labels = mat['y']
		data = torch.from_numpy(data)
		self.labels = torch.from_numpy(labels).squeeze(1)#.unsqueeze(1)
		self.data = data.transpose(0,3).transpose(1,2)

		#if oracle:
		self.labels = self.labels.unsqueeze(1)
		#svhn has 10 means 0, so replace the labels
		idx = torch.where(self.labels == 10)
		self.labels[idx] = 0
		flag = torch.tensor([0]*labels.shape[0]).unsqueeze(1)
		self.labels = torch.cat((self.labels, flag), 1)
		#print(self.data[0,:,:,:])
		self.means = (self.data / 255.0).mean(axis=(0,2,3))#.unsqueeze(1).unsqueeze(2)
		self.stds = (self.data / 255.0).std(axis=(0,2,3))#.unsqueeze(1).unsqueeze(2)

		#print(self.means, self.stds)

		self.normalize = transforms.Normalize(mean = self.means,
				  std = self.stds)

	def __getitem__(self, index):
		return self.normalize(self.data[index] / 255.0), self.labels[index]

	def __len__(self):
		return self.labels.shape[0]


	
class DMNIST(Dataset):
	def __init__(self, data, labels, data_aug=False):
		self.data = data.unsqueeze(1)
		self.labels = labels

		#since MNIST images are 28x28, we resize them to 32x32
		resize = transforms.Compose([transforms.Resize(32),])

		self.data = resize(self.data)

		self.means = (self.data / 255.0).mean(axis=(0,2,3))
		self.stds = (self.data / 255.0).std(axis=(0,2,3))

		normalize = transforms.Normalize(mean = self.means,
				  std = self.stds)

		# self.severity = severity
		# normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
		# 		  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
		#self.normalize = transforms.Normalize(means,stds)
		if data_aug:
			self.transform = transforms.Compose([
				#transforms.Resize(32), 
				transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
												  (4, 4, 4, 4), mode='reflect').squeeze()),
				transforms.ToPILImage(),
				transforms.RandomCrop(32),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			])
		else:
				self.transform = transforms.Compose([
					#transforms.Resize(32),
					normalize,
			])

	def normalize(self, image):
		return self.transform(image)


	@classmethod
	def read(cls, test=False, data_aug=False):
		if not test:
			(mnist_train_data, mnist_train_labels), (mnist_val_data, mnist_val_labels) = DMNIST.mnist(test=test)

			(a_train_data, a_train_labels), (a_val_data, a_val_labels) = DMNIST.ambiguous(test=test)

			train_data = torch.cat((mnist_train_data, a_train_data), 0)
			train_labels = torch.cat((mnist_train_labels, a_train_labels), 0)

			val_data = torch.cat((mnist_val_data, a_val_data), 0)
			val_labels = torch.cat((mnist_val_labels, a_val_labels), 0)
			return cls(train_data, train_labels, data_aug=data_aug), cls(val_data, val_labels, data_aug=False)
		else:
			mnist = DMNIST.mnist(test=True)
			ambiguous = DMNIST.ambiguous(test=True)

			return cls(mnist[0], mnist[1]), cls(ambiguous[0], ambiguous[1])


	@staticmethod
	def mnist(test=False):
		if not test:
			mnist = datasets.MNIST('../data', train=True, download=True)
			data = mnist.data
			labels = mnist.targets.unsqueeze(1)
			flag = torch.tensor([1]*labels.shape[0]).unsqueeze(1)
			labels = torch.cat((labels, flag), 1)

			(train_data, train_labels), (val_data, val_labels) = DMNIST.split(data, labels)
			return (train_data, train_labels), (val_data, val_labels)
		else:
			mnist = datasets.MNIST('../data', train=False, download=True)
			data = mnist.data
			labels = mnist.targets.unsqueeze(1)
			flag = torch.tensor([1]*labels.shape[0]).unsqueeze(1)
			labels = torch.cat((labels, flag), 1)

			print(data.shape, labels.shape)

			return (data, labels)

	@staticmethod
	def ambiguous(test=False):
		if not test:
			amnist = ddu_dirty_mnist.AmbiguousMNIST("../data", train=True, download=True)
			a_labels = amnist.targets.view(6000,10).numpy()
			temp = a_labels - np.expand_dims(a_labels[:,0], axis=1)
			idx = torch.tensor(np.argwhere(np.all(temp[:, ...] == 0, axis=1))).squeeze()
			data = amnist.data.squeeze(1)[idx]
			labels = amnist.targets[idx].unsqueeze(1)
			flag = torch.tensor([0]*labels.shape[0]).unsqueeze(1)
			labels = torch.cat((labels, flag), 1)
			(train_data, train_labels), (val_data, val_labels) = DMNIST.split(data, labels)
			return (train_data, train_labels), (val_data, val_labels)
		else:
			amnist = ddu_dirty_mnist.AmbiguousMNIST("../data", train=False, download=True)
			a_labels = amnist.targets.view(6000,10).numpy()
			unique_labels = np.unique(a_labels, axis=1)

			#print(unique_labels)

			temp = unique_labels - np.expand_dims(unique_labels[:,0], axis=1)
			idx = torch.tensor(np.argwhere(np.all(temp[:, ...] == 0, axis=1))).squeeze()
			idx_ = sorted(np.setdiff1d(range(a_labels.shape[0]), idx.numpy(), assume_unique=True).tolist())


			print("Ambiguous Samples with non-unique labels are {}".format(len(idx_)))

			non_unique_labels = unique_labels[idx_]
			rng = np.random.default_rng()
			labels = rng.choice(non_unique_labels, axis=1) #labels shape = (6000,)
			idx = list(range(0, 60000, 10))
			data = amnist.data.squeeze(1)[idx][idx_]
			flag = torch.tensor([0]*labels.shape[0]).unsqueeze(1)
			labels = torch.cat((torch.from_numpy(labels).view(-1,1), flag), dim=1)

			print(labels.shape, data.shape)

			return (data, labels)


	@staticmethod
	def split(data, labels):
		idx = torch.randperm(data.shape[0])
		data = data[idx].view(data.size())
		labels = labels[idx].view(labels.size())

		train_size = int(0.90 * len(data))
		train_data = data[:train_size,:,:]
		train_labels = labels[:train_size,:]

		val_data = data[train_size:,:,:]
		print(train_data.shape, val_data.shape)
		val_labels = labels[train_size:,:]
		return (train_data, train_labels), (val_data, val_labels)

	def __getitem__(self, index):
		return self.normalize(self.data[index] / 255.0).transpose(1,2), self.labels[index]

	def __len__(self):
		return self.labels.shape[0]


if __name__ == "__main__":
	
	# train, val = cifar.read(severity=1, slice_=-1, test=False, only_id=True)
	# val = SVHN()
	# train = SVHN()
	# print(torch.where(train.labels[:,0] == 10))

	train, val = cifar.read(test=False, only_id=True, data_aug=True, split_train_set=True, train_proportion=0.8) #DMNIST.read(test=True, data_aug=True)

	print(type(train), type(val))
	print(train.data.shape, val.data.shape)

	def test(data):
		dl = DataLoader(data, batch_size=1024, shuffle=True)
		for batch in dl:
			img, label = batch
			print(label)
			print(label.shape)
			print(img.shape)
			print("Batch mean ", img.mean(dim=[0,2,3]))
			print("Batch std ", img.std(dim=[0,2,3]))
			break
	test(train)
	test(val)