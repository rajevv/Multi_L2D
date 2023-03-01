import argparse
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from validation_expert_model import *

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default="ConvMixer")

parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--scale', default=0.75, type=float)
parser.add_argument('--reprob', default=0.25, type=float)
parser.add_argument('--ra-m', default=8, type=int)
parser.add_argument('--ra-n', default=1, type=int)
parser.add_argument('--jitter', default=0.1, type=float)

parser.add_argument('--hdim', default=256, type=int)
parser.add_argument('--depth', default=8, type=int)
parser.add_argument('--psize', default=2, type=int)
parser.add_argument('--conv-ks', default=5, type=int)

parser.add_argument('--wd', default=0.01, type=float)
parser.add_argument('--clip-norm', action='store_true')
parser.add_argument('--epochs', default=75, type=int)
parser.add_argument('--lr-max', default=0.01, type=float)
parser.add_argument('--workers', default=2, type=int)

args = parser.parse_args()


class Residual(nn.Module):
	def __init__(self, fn):
		super().__init__()
		self.fn = fn

	def forward(self, x):
		return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=5, patch_size=2, n_classes=10):
	return nn.Sequential(
		nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
		nn.GELU(),
		nn.BatchNorm2d(dim),
		*[nn.Sequential(
				Residual(nn.Sequential(
					nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
					nn.GELU(),
					nn.BatchNorm2d(dim)
				)),
				nn.Conv2d(dim, dim, kernel_size=1),
				nn.GELU(),
				nn.BatchNorm2d(dim)
		) for i in range(depth)],
		nn.AdaptiveAvgPool2d((1,1)),
		nn.Flatten(),
		nn.Linear(dim, n_classes)
	)


if __name__ == "__main__":
	import os

	from data_utils import *
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cudnn.benchmark = True

	print(device)

	ckp_path = './MLP_Mixer_model/'
	os.makedirs(ckp_path, exist_ok=True)

	train_data, val_data, _ = ham10000_expert.read(data_aug=True)

	batch_size = 64

	# cifar10_mean = (0.4914, 0.4822, 0.4465)
	# cifar10_std = (0.2471, 0.2435, 0.2616)

	# train_transform = transforms.Compose([
	#     transforms.RandomResizedCrop(32, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),
	#     transforms.RandomHorizontalFlip(p=0.5),
	#     transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
	#     transforms.ColorJitter(args.jitter, args.jitter, args.jitter),
	#     transforms.ToTensor(),
	#     transforms.Normalize(cifar10_mean, cifar10_std),
	#     transforms.RandomErasing(p=args.reprob)
	# ])

	# test_transform = transforms.Compose([
	#     transforms.ToTensor(),
	#     transforms.Normalize(cifar10_mean, cifar10_std)
	# ])

	# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	#                                         download=True, transform=train_transform)
	# trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
	#                                           shuffle=True, num_workers=args.workers)

	# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
	#                                        download=True, transform=test_transform)
	# testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
	#                                          shuffle=False, num_workers=args.workers)


	kwargs = {'num_workers': 0, 'pin_memory': True}
	trainloader = torch.utils.data.DataLoader(train_data,
											   batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
	testloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
	
	model = ConvMixer(args.hdim, args.depth, patch_size=args.psize, kernel_size=args.conv_ks, n_classes=7)
	model = nn.DataParallel(model).cuda()


	lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], 
									  [0, args.lr_max, args.lr_max/20.0, 0])[0]

	opt = optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.wd)
	criterion = nn.CrossEntropyLoss()
	scaler = torch.cuda.amp.GradScaler()


	for epoch in range(args.epochs):
		start = time.time()
		train_loss, train_acc, n = 0, 0, 0
		for i, (X, y, _) in enumerate(trainloader):
			model.train()
			X, y = X.float().cuda(), y.cuda()

			lr = lr_schedule(epoch + (i + 1)/len(trainloader))
			opt.param_groups[0].update(lr=lr)

			opt.zero_grad()
			with torch.cuda.amp.autocast():
				output = model(X)
				loss = criterion(output, y)

			scaler.scale(loss).backward()
			if args.clip_norm:
				scaler.unscale_(opt)
				nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			scaler.step(opt)
			scaler.update()
			
			train_loss += loss.item() * y.size(0)
			train_acc += (output.max(1)[1] == y).sum().item()
			n += y.size(0)
			
		model.eval()
		best_test_acc = -1.0
		test_acc, m = 0, 0
		with torch.no_grad():
			for i, (X, y, _) in enumerate(testloader):
				X, y = X.float().cuda(), y.cuda()
				with torch.cuda.amp.autocast():
					output = model(X)
				test_acc += (output.max(1)[1] == y).sum().item()
				m += y.size(0)

			if test_acc >= best_test_acc:
				torch.save(model, ckp_path + 'm_convMixer_expert_v2')
				best_test_acc = test_acc
				print('updated the model', flush=True)
				patience = 0
				# test the model

				_ , _ , test_data = ham10000_expert.read(data_aug=True)
				testloader = torch.utils.data.DataLoader(test_data,
															   batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

				#evaluationMLPMixer(testloader)
				print("evaluation on test data...")
				evaluationConvMixer(testloader, 'm_convMixer_expert_v2')

		print(f'[{args.name}] Epoch: {epoch} | Train Acc: {train_acc/n:.4f}, Test Acc: {test_acc/m:.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}')