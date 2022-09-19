import torch
from torch import nn
from torch.nn import functional as F
import wandb
import torch.backends.cudnn as cudnn


class MLP(nn.Module):
	def __init__(self, num_features, expansion_factor, dropout):
		super().__init__()
		num_hidden = num_features * expansion_factor
		self.fc1 = nn.Linear(num_features, num_hidden)
		self.dropout1 = nn.Dropout(dropout)
		self.fc2 = nn.Linear(num_hidden, num_features)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, x):
		x = self.dropout1(F.gelu(self.fc1(x)))
		x = self.dropout2(self.fc2(x))
		return x


class TokenMixer(nn.Module):
	def __init__(self, num_features, num_patches, expansion_factor, dropout):
		super().__init__()
		self.norm = nn.LayerNorm(num_features)
		self.mlp = MLP(num_patches, expansion_factor, dropout)

	def forward(self, x):
		# x.shape == (batch_size, num_patches, num_features)
		residual = x
		x = self.norm(x)
		x = x.transpose(1, 2)
		# x.shape == (batch_size, num_features, num_patches)
		x = self.mlp(x)
		x = x.transpose(1, 2)
		# x.shape == (batch_size, num_patches, num_features)
		out = x + residual
		return out


class ChannelMixer(nn.Module):
	def __init__(self, num_features, num_patches, expansion_factor, dropout):
		super().__init__()
		self.norm = nn.LayerNorm(num_features)
		self.mlp = MLP(num_features, expansion_factor, dropout)

	def forward(self, x):
		# x.shape == (batch_size, num_patches, num_features)
		residual = x
		x = self.norm(x)
		x = self.mlp(x)
		# x.shape == (batch_size, num_patches, num_features)
		out = x + residual
		return out


class MixerLayer(nn.Module):
	def __init__(self, num_features, num_patches, expansion_factor, dropout):
		super().__init__()
		self.token_mixer = TokenMixer(
			num_patches, num_features, expansion_factor, dropout
		)
		self.channel_mixer = ChannelMixer(
			num_patches, num_features, expansion_factor, dropout
		)

	def forward(self, x):
		# x.shape == (batch_size, num_patches, num_features)
		x = self.token_mixer(x)
		x = self.channel_mixer(x)
		# x.shape == (batch_size, num_patches, num_features)
		return x


def check_sizes(image_size, patch_size):
	sqrt_num_patches, remainder = divmod(image_size, patch_size)
	assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
	num_patches = sqrt_num_patches ** 2
	return num_patches


class MLPMixer(nn.Module):
	def __init__(
		self,
		image_size=256,
		patch_size=16,
		in_channels=3,
		num_features=128,
		expansion_factor=2,
		num_layers=8,
		num_classes=10,
		dropout=0.5,
	):
		num_patches = check_sizes(image_size, patch_size)
		super().__init__()
		# per-patch fully-connected is equivalent to strided conv2d
		self.patcher = nn.Conv2d(
			in_channels, num_features, kernel_size=patch_size, stride=patch_size
		)
		self.mixers = nn.Sequential(
			*[
				MixerLayer(num_patches, num_features, expansion_factor, dropout)
				for _ in range(num_layers)
			]
		)
		self.classifier = nn.Linear(num_features+3, num_classes)

	def forward(self, x, z):
		patches = self.patcher(x)
		batch_size, num_features, _, _ = patches.shape
		patches = patches.permute(0, 2, 3, 1)
		patches = patches.view(batch_size, -1, num_features)
		# patches.shape == (batch_size, num_patches, num_features)
		embedding = self.mixers(patches)
		# embedding.shape == (batch_size, num_patches, num_features)
		embedding = embedding.mean(dim=1)
		embedding = torch.cat([embedding, z], dim=1)
		logits = self.classifier(embedding)
		return logits

class Metrics(object):
    def __init__(self):
        pass

    def accuracy(self, pred, actual):
        acc = torch.sum((torch.argmax(pred, dim=1) == actual).float())
        acc /= pred.shape[0]
        return acc


def train(config, device, model, trainloader, validloader, ckp_path):  
	model.to(device)
	num_batches = len(trainloader)
	warmup_iters = 5*num_batches
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(),lr=config['lrate'], weight_decay=config['weight_decay'])
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_batches * config['epochs'])
	metrics = Metrics()
	train_losses = []
	val_losses = []
	best_val_loss = 1000
	max_patience = 20
	patience = 0
	eps = 1e-4
	iter = 0
	for epoch in range(config['epochs']):
		print('----- epoch:',epoch, '-----',  flush=True)
		epoch_loss = 0
		val_epoch_loss = 0
		for i, batch in enumerate(trainloader):
			## for warmp-ups to avoid overfitting
			if iter < warmup_iters:
				lr = config['lrate']*float(iter) / warmup_iters
				print(iter, lr)
				for param_group in optimizer.param_groups:
					param_group['lr'] = lr
			
			X, Y, Z = batch

			X = X.float().to(device)
			Y = Y.to(device)
			Z = Z.float().to(device)

			logits = model(X, Z)
			optimizer.zero_grad()
			loss = criterion(logits, Y)
			loss.backward()
			optimizer.step()
			## initiate scheduler only after warmup period
			if not iter < warmup_iters:
				scheduler.step()
			epoch_loss += float(loss.item())
			iter+=1
		print('[{}]/[{}] Train loss: {}'.format(epoch, config['epochs'], epoch_loss/ num_batches),  flush=True)
		train_loss = epoch_loss / num_batches
		with torch.no_grad():
			vloss = []
			vacc = []
			for i, batch in enumerate(validloader):
				val_X, val_Y, val_Z = batch
				val_X = val_X.to(device)
				val_Y = val_Y.to(device)
				val_Z = val_Z.to(device)
				val_logits = model(val_X.float(), val_Z.float())
				val_loss = criterion(val_logits, val_Y)
				vloss.append(val_loss.item())
				vacc.append(metrics.accuracy(val_logits, val_Y).item())
			
			val_loss = np.average(vloss)
			val_acc = np.average(vacc)
			print('Validation loss: {}, Validation accuracy: {}'.format(float(val_loss), float(val_acc)), flush=True)
			
			if val_loss < best_val_loss:
				torch.save(model.state_dict(), ckp_path + 'm_expert')
				best_val_loss = val_loss
				print('updated the model', flush=True)
				patience = 0
			else:
				patience += 1
			if patience > max_patience:
				print('no progress for 10 epochs... stopping training', flush=True)
				break
		wandb.log({"training_loss": train_loss, "validation_loss": val_loss})
		print('\n')



if __name__ == "__main__":
	from data_utils import *
	import os
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cudnn.benchmark = True

	print(device)

	ckp_path = './Models/'
	os.makedirs(ckp_path, exist_ok=True)

	train_data, val_data, _ = ham10000_expert.read(data_aug=True)

	batch_size = 1024
	config = {
					"setting": "expert model default",
					"epochs" : 200,
					"patience": 50,
					"batch_size": batch_size,
					"lrate": 0.001,
					"weight_decay":5e-4
				}

	model = MLPMixer(image_size=224, patch_size=16, in_channels=3, num_features=128, expansion_factor=2, num_layers=8, num_classes=7)
	kwargs = {'num_workers': 0, 'pin_memory': True}
	trainloader = torch.utils.data.DataLoader(train_data,
											   batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
	validloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
	run = wandb.init(project="ham10000 expert", reinit=True, config=config, entity="aritzz")
	train(config, device, model, trainloader, validloader, ckp_path)
	run.finish()
