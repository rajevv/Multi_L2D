# %%
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split
import csv
import numpy as np
from tqdm import tqdm 

preprocess = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


ham10000_path = './HAM10000/'
images_path = ham10000_path + "HAM10000_images/" 

with open(ham10000_path + 'HAM10000_metadata.csv', 'r') as f:
	metadata = [{k:v for k,v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]

new_data = []

all_images = os.listdir(images_path)


image_data = []
labels = []
diagnosis_type = []
age = []
gender = []

label_dict = {'bkl':0, 'df':1, 'mel':2, 'nv':3, 'vasc':4, 'akiec':5, 'bcc':6}
diagnosis_type_dict = {'histo': 0, 'follow_up': 1, 'consensus': 2, 'confocal': 3}
gender_dict = {'male': 0, 'female': 1, 'unknown': 2}

for image in tqdm(metadata):
	if image['image_id'] + '.jpg' not in all_images:
		continue
	if image['age'] == '':
		continue
	img_data = Image.open(images_path + image['image_id']+'.jpg')
	image_data.append(preprocess(img_data).unsqueeze(0))
	labels.append(label_dict[image['dx']])
	diagnosis_type.append(diagnosis_type_dict[image['dx_type']])
	age.append(float(image['age']))
	gender.append(gender_dict[image['sex']])


image_d = torch.cat(image_data, dim=0)

train_idx, test_idx = train_test_split(
										np.arange(image_d.shape[0]),
										test_size = 0.15,
										shuffle = True,
										stratify = np.array(labels)
										)
			


train_data = image_d[train_idx]
train_labels = np.array(labels)[train_idx]
train_diagnosis_type = np.array(diagnosis_type)[train_idx]
train_age = np.array(age)[train_idx]
train_gender = np.array(gender)[train_idx]


test_data = image_d[test_idx]
test_labels = np.array(labels)[test_idx]
test_diagnosis_type = np.array(diagnosis_type)[test_idx]
test_age = np.array(age)[test_idx]
test_gender = np.array(gender)[test_idx]


train_idx_, val_idx = train_test_split(
										np.arange(train_data.shape[0]),
										test_size = 0.25,
										shuffle = True,
										stratify = train_labels
										)


train_data_ = train_data[train_idx_]
train_labels_ = train_labels[train_idx_]
train_diagnosis_type_ = train_diagnosis_type[train_idx_]
train_age_ = train_age[train_idx_]
train_gender_ = train_gender[train_idx_]

val_data = train_data[val_idx]
val_labels = train_labels[val_idx]
val_diagnosis_type = train_diagnosis_type[val_idx]
val_age = train_age[val_idx]
val_gender = train_gender[val_idx]


train = {'data': train_data_, 
		'labels': train_labels_, 
		'diagnosis_type': train_diagnosis_type_,
		'age': train_age_,
		'gender': train_gender_
		}

val = {'data': val_data, 
		'labels': val_labels, 
		'diagnosis_type': val_diagnosis_type,
		'age': val_age,
		'gender': val_gender
		}       

test = {'data': test_data, 
		'labels': torch.from_numpy(test_labels), 
		'diagnosis_type': torch.from_numpy(test_diagnosis_type),
		'age': torch.from_numpy(test_age),
		'gender': torch.from_numpy(test_gender)
		}

import os
os.makedirs('./Data', exist_ok=True)

torch.save(train, './Data/train_data.pt')
torch.save(val, './Data/validation_data.pt')
torch.save(test, './Data/test_data.pt')