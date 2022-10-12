import os
import pandas as pd
import numpy as np
import skimage
from skimage import io
import csv
from skimage.transform import rescale, resize, downscale_local_mean
import torch
import torch.nn as nn
import copy

# entire dataset size = 61577
'''
The dataset is downloaded from https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data
put the training_solutions_rev1.csv and images_training_rev1 folder in this directory and then run it using python3 prepare_data.py
'''
num_samples = 10000

X = np.zeros((num_samples,3,224,224),dtype='float')
Y = np.zeros(num_samples,dtype=int)
c = np.zeros((num_samples,2))
hpred = np.zeros(num_samples,dtype=int)

table = csv.reader(open('training_solutions_rev1.csv'))
next(table)

idx = 0

for dirname, _, filenames in os.walk('images_training_rev1'):
    for i,tuple in enumerate(zip(sorted(filenames)[:num_samples],table)):
        filename = tuple[0]
        row = tuple[1]
        print(idx)
        if idx == num_samples:
            break
        file = os.path.join(dirname, filename)
        from PIL import Image
        from torchvision import transforms
        input_image = Image.open(file)
        print(np.max(np.asarray(input_image)),np.min(np.asarray(input_image)))
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = preprocess(input_image)


        image_id = filename[:-4]
        eps = 0.1
        probs = [float(row[1]),float(row[2])]
        epsilon = 0.001
        for j,prob in enumerate(probs):
            if probs[j] < epsilon:
                probs[j] = epsilon
        if np.argmax(np.array(probs))==2:
            continue
        X[idx] = img
        Y[idx] = np.argmax(np.array(probs))
        p0 = probs[0]
        p1 = probs[1]
        hpred[idx] = np.random.choice(2,1,p=[p0/(p0+p1),p1/(p0+p1)])
        print(probs[0]/(probs[0]+probs[1]),probs[1]/(probs[0]+probs[1]))
        if probs[0]>probs[1]:
            probs[0]-=eps
            probs[1]+=eps
        else:
            probs[1]-=eps
            probs[0]+=eps
        assert(row[0] == image_id)
        c[idx] = np.array(probs)
        
        print(Y[idx],hpred[idx],c[idx],[p0/(p0+p1),p1/(p0+p1)])
        print('------')
        idx += 1


print(np.mean(Y==hpred))
print(X.shape)
print(np.sum([1 if y==0 else 0 for y in Y]))
print(np.sum([1 if y==1 else 0 for y in Y]))
print(np.sum([1 if y==2 else 0 for y in Y]))

import random
frac = 0.6
num_train = int(frac*num_samples)
num_test = int((num_samples - num_train)/2)
num_val = num_samples - (num_train + num_test)

indices = np.arange(num_samples)
random.shuffle(indices)
indices_train = indices[:num_train]
indices_val = indices[num_train:num_train+num_val]
indices_test = indices[num_train+num_val:]

x_train = X[indices_train]
y_train = Y[indices_train]
c_train = c[indices_train]
hpred_train = hpred[indices_train]

x_val = X[indices_val]
y_val = Y[indices_val]
c_val = c[indices_val]
hpred_val = hpred[indices_val]


x_test = X[indices_test]
y_test = Y[indices_test]
c_test = c[indices_test]
hpred_test = hpred[indices_test]

data = {}
data['test'] = {}
data['val'] = {}

data['c'] = {}
data['test']['c'] = {}
data['val']['c'] = {}

data['X'] = x_train
data['Y'] = y_train
data['c']['0.0'] = c_train
data['hpred'] = hpred_train

data['test']['X'] = x_test
data['test']['Y'] = y_test
data['test']['c']['0.0'] = c_test
data['test']['hpred'] = hpred_test

data['val']['X'] = x_val
data['val']['Y'] = y_val
data['val']['c']['0.0'] = c_val
data['val']['hpred'] = hpred_val

loss_func = torch.nn.NLLLoss(reduction='none')

hscores = torch.log(torch.from_numpy(copy.deepcopy(data['c']['0.0'])).float())
hloss = loss_func(hscores, torch.from_numpy(data['Y']).long())

hconf, _ = torch.max(hscores, axis=1)

val_hscores = torch.log(torch.from_numpy(copy.deepcopy(data['val']['c']['0.0'])).float())
val_hloss = loss_func(val_hscores, torch.from_numpy(data['val']['Y']).long())
val_hconf, _ = torch.max(val_hscores, axis=1)


test_hscores = torch.log(torch.from_numpy(copy.deepcopy(data['test']['c']['0.0'])).float())
test_hloss = loss_func(test_hscores, torch.from_numpy(data['test']['Y']).long())
test_hconf, _ = torch.max(test_hscores, axis=1)

data['hloss'] = hloss
data['val']['hloss'] = val_hloss
data['test']['hloss'] = test_hloss
data['human_is_correct'] = np.array([hpred==y for hpred,y in zip(data['hpred'],data['Y'])])

data['hconf'] = hconf
data['val']['hconf'] = val_hconf
data['test']['hconf'] = test_hconf
data['val']['human_is_correct'] = np.array([hpred==y for hpred,y in zip(data['val']['hpred'],data['val']['Y'])])

data['hscores'] = hscores
data['val']['hscores'] = val_hscores
data['test']['hscores'] = test_hscores
data['test']['human_is_correct'] = np.array([hpred==y for hpred,y in zip(data['test']['hpred'],data['test']['Y'])])

import pickle
with open('galaxy_data.pkl','wb') as f:
    pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)