import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import classification_report
from expert_model import MLPMixer
from data_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def evaluation(model, testloader):
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





_ , _ , test_data = ham10000_expert.read(data_aug=True)

model = MLPMixer(image_size=224, patch_size=16, in_channels=3, num_features=128, expansion_factor=2, num_layers=8, num_classes=7)
model.load_state_dict(torch.load('./Models/m_expert'))
model.eval()



batch_size=1024
kwargs = {'num_workers': 0, 'pin_memory': True}
testloader = torch.utils.data.DataLoader(test_data,
											   batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

evaluation(model, testloader)