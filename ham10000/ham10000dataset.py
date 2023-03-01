from __future__ import division

import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

ham10000_label_dict = {'bkl': 0, 'df': 1, 'mel': 2,
                       'nv': 3, 'vasc': 4, 'akiec': 5, 'bcc': 6}
mal_dx = ["mel", "bcc", "akiec"]
ben_dx = ["nv", "bkl", "df", "vasc"]

# regular dataset with images and labels


class ham10000_defer(Dataset):
    def __init__(self, img_data, labels, data_aug=False):
        super(ham10000_defer).__init__()
        self.data = img_data
        self.labels = labels
        self.data_aug = data_aug

        if self.data_aug:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

    @classmethod
    def read(cls, path='./Data/', data_aug=False):
        train = torch.load(path + 'train_data.pt')
        val = torch.load(path + 'validation_data.pt')
        test = torch.load(path + 'test_data.pt')

        return cls(train['data'], train['labels'], data_aug=data_aug), cls(val['data'], val['labels']), cls(test['data'], test['labels'])

    def __getitem__(self, index):
        if self.data_aug:
            return self.transform(self.data[index]), self.labels[index]
        else:
            return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]

# expert dataset with image metadata to be used by the expert


class ham10000_expert(Dataset):
    def __init__(self, img_data, labels, diagnosis_type, age, gender, data_aug=False):
        super(ham10000_expert).__init__()
        self.data = img_data
        self.labels = labels
        self.diagnosis_type = diagnosis_type
        self.age = age
        self.gender = gender
        self.data_aug = data_aug

        if self.data_aug:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

    @classmethod
    def read(cls, path='./Data/', data_aug=False):
        train = torch.load(path + 'train_data.pt')
        val = torch.load(path + 'validation_data.pt')
        test = torch.load(path + 'test_data.pt')

        return cls(train['data'], train['labels'], train['diagnosis_type'], train['age'], train['gender']),\
            cls(val['data'], val['labels'], val['diagnosis_type'], val['age'], val['gender']),\
            cls(test['data'], test['labels'],
                test['diagnosis_type'], test['age'], test['gender'])

    def __getitem__(self, index):
        if not self.data_aug:
            return self.data[index], self.labels[index], torch.tensor([self.diagnosis_type[index], self.age[index], self.gender[index]])
        else:
            return self.transform(self.data[index]), self.labels[index], torch.tensor([self.diagnosis_type[index], self.age[index], self.gender[index]])

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":

    train, val, test = ham10000_defer.read(data_aug=True)
    print(train.data.shape, val.data.shape)

    print("HAM10000_defer")

    def check(data):
        dl = DataLoader(data, batch_size=1024, shuffle=True)
        for batch in dl:
            img, label = batch
            print(label)
            print(type(label))
            print(label.shape)
            print(img.shape)
            break
    check(train)
    check(val)
    check(test)

    print("HAM10000_expert")

    train, val, test = ham10000_expert.read(data_aug=True)
    print(train.data.shape, val.data.shape)

    def check(data):
        dl = DataLoader(data, batch_size=1024, shuffle=True)
        for batch in dl:
            img, label, z = batch
            print(z.shape)
            print(label)
            print(type(label))
            print(label.shape)
            print(img.shape)
            break
    check(train)
    check(val)
    check(test)
