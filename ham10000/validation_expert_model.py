import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import classification_report
from expert_model import MLPMixer
from conv_mixer_model import *
from data_utils import *
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
parser.add_argument('--epochs', default=25, type=int)
parser.add_argument('--lr-max', default=0.01, type=float)
parser.add_argument('--workers', default=2, type=int)

args = parser.parse_args()


def evaluation_MLPMixer(testloader):
	model = MLPMixer(image_size=224, patch_size=16, in_channels=3, num_features=128, expansion_factor=2, num_layers=8, num_classes=7)
	model.load_state_dict(torch.load('./Models/m_expert'))
	model.eval()
	model.to(device)

	prediction = []
	actual = []
	for i, batch in enumerate(testloader):
		X, Y, Z = batch
		X = X.float().to(device)
		Z = Z.float().to(device)

		logits = model(X, Z)

		pred = torch.argmax(logits, dim=1)

		prediction.extend(pred.cpu().tolist())
		actual.extend(Y.tolist())

	print(len(prediction), len(actual), flush=True)

	label_dict = {'bkl':0, 'df':1, 'mel':2, 'nv':3, 'vasc':4, 'akiec':5, 'bcc':6}
	target_names = ['bkl', 'df', 'mel', 'nv', 'vasc', 'akiec', 'bcc']
	print(classification_report(actual, prediction, target_names=target_names), flush=True)


def evaluationConvMixer(testloader, model_name):
	model = torch.load('./Models/' + model_name) #ConvMixer(args.hdim, args.depth, patch_size=args.psize, kernel_size=args.conv_ks, n_classes=7)
	#model.load_state_dict(torch.load('./Models/' + model_name), strict=False)
	model = nn.DataParallel(model).to(device)
	model.eval()
	
	pred = []
	actual = []
	test_acc, m = 0.0, 0.0
	with torch.no_grad():
		for i, (X, y, _) in enumerate(testloader):
			X, y = X.float().to(device), y.to(device)
			with torch.cuda.amp.autocast():
				output = model(X)
			test_acc += (output.max(1)[1] == y).sum().item()
			m += y.size(0)
			pred.extend(output.max(1)[1].cpu().tolist())
			actual.extend(y.cpu().tolist())

	print(test_acc/m, len(pred), len(actual), flush=True)

	label_dict = {'bkl':0, 'df':1, 'mel':2, 'nv':3, 'vasc':4, 'akiec':5, 'bcc':6}
	target_names = ['bkl', 'df', 'mel', 'nv', 'vasc', 'akiec', 'bcc']
	print(classification_report(actual, pred, target_names=target_names), flush=True)


		



if __name__ == "__main__":
	_ , _ , test_data = ham10000_expert.read(data_aug=True)

	batch_size=64
	kwargs = {'num_workers': 0, 'pin_memory': True}
	testloader = torch.utils.data.DataLoader(test_data,
												   batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

	#evaluationMLPMixer(testloader)
	# print("25 epochs...")
	# evaluationConvMixer(testloader, 'm_convMixer_expert')

	print("75 epochs...")
	evaluationConvMixer(testloader, 'm_convMixer_expert_v2')

    