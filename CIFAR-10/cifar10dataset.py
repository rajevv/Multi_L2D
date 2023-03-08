from __future__ import division

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from scipy import io
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class cifar(Dataset):
    def __init__(self, data, labels, split_train_set=False, train_proportion=1.0, data_aug=False):
        super(cifar).__init__()
        self.data = data
        self.labels = labels
        #print("Dataset Size ", self.data.shape, self.labels.shape)

        if split_train_set:

            train_idx, _ = train_test_split(
                np.arange(self.labels.shape[0]),
                test_size=1.0 - train_proportion,
                shuffle=True,
                stratify=self.labels[:, 0].numpy()
            )

            self.data = self.data[train_idx]
            self.labels = self.labels[train_idx]
            print("Dataset Size after splitting ",
                  self.data.shape, self.labels.shape)

        # .unsqueeze(1).unsqueeze(2)
        self.means = (self.data / 255.0).mean(axis=(0, 2, 3))
        # .unsqueeze(1).unsqueeze(2)
        self.stds = (self.data / 255.0).std(axis=(0, 2, 3))

        normalize = transforms.Normalize(mean=self.means,
                                         std=self.stds)

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
    def read(cls, severity=0, slice_=-1, test=False, data_aug=False, entropy=False, mix=False, only_id=False, only_ood=False, split_train_set=False, train_proportion=1.0):

        # This will allow to have always same test results! Careful!
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

        if not test:
            if only_ood:
                (ood_train_data, ood_train_labels), (ood_val_data,
                                                     ood_val_labels) = cifar.OOD(severity, slice_, test)
                return cls(ood_train_data, ood_train_labels, data_aug=data_aug), cls(ood_val_data, ood_val_labels, data_aug=False)

            if only_id:
                (id_train_data, id_train_labels), (id_val_data,
                                                   id_val_labels) = cifar.ID(test)
                return cls(id_train_data, id_train_labels, data_aug=data_aug, split_train_set=split_train_set, train_proportion=train_proportion), cls(id_val_data, id_val_labels, data_aug=False)

            if entropy:
                (ood_train_data, ood_train_labels), (ood_val_data,
                                                     ood_val_labels) = cifar.OOD(severity, slice_, test)
                (id_train_data, id_train_labels), (id_val_data,
                                                   id_val_labels) = cifar.ID(test)
                return cls(id_val_data, id_val_labels, data_aug=False), cls(ood_val_data, ood_val_labels, data_aug=False)

            if mix:
                (ood_train_data, ood_train_labels), (ood_val_data,
                                                     ood_val_labels) = cifar.OOD(severity, slice_, test)
                (id_train_data, id_train_labels), (id_val_data,
                                                   id_val_labels) = cifar.ID(test)

                train_data = torch.cat((ood_train_data, id_train_data), 0)
                train_labels = torch.cat(
                    (ood_train_labels, id_train_labels), 0)

                val_data = torch.cat((ood_val_data, id_val_data), 0)
                val_labels = torch.cat((ood_val_labels, id_val_labels), 0)
                return cls(train_data, train_labels, data_aug=data_aug), cls(val_data, val_labels, data_aug=False)

        else:
            ood_data, ood_labels = cifar.OOD(severity, slice_, test)

            id_data, id_labels = cifar.ID(test)

            return cls(ood_data, ood_labels, data_aug=False), cls(id_data, id_labels, data_aug=False)

    @staticmethod
    def OOD(severity, slice_, test):
        corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost',
                       'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur',
                       'pixelate', 'shot_noise', 'snow', 'zoom_blur'][:slice_]

        path = '../data/CIFAR-10-C/'
        data = []
        labels = []
        lbl_file = np.load(path + 'labels.npy')
        for corruption in corruptions:
            # messy code but we are assuming we always train with the 0th perturbations
            data_ = np.load(path + corruption + '.npy')
            if severity != 0 and test == True:
                data.append(
                    data_[10000*severity: (10000*severity+10000), :, :, :])
                labels.append(lbl_file[10000*severity: (10000*severity+10000)])
            elif severity == 0 and test == True:
                data.append(data_[8000:10000, :, :, :])
                labels.append(lbl_file[8000: 10000])
            else:
                data.append(
                    data_[10000*severity: (10000*severity+8000), :, :, :])
                labels.append(lbl_file[10000*severity: (10000*severity+8000)])

        data = torch.from_numpy(np.concatenate(data)).float().transpose(1, 3)
        labels = torch.from_numpy(np.concatenate(labels)).unsqueeze(1)
        flag = torch.tensor([0]*labels.shape[0]).unsqueeze(1)
        labels = torch.cat((labels, flag), 1)

        if not test:
            idx = torch.randperm(data.shape[0])
            data = data[idx].view(data.size())
            labels = labels[idx].view(labels.size())

            train_size = int(0.90 * len(data))
            train_data = data[:train_size, :, :, :]
            train_labels = labels[:train_size, :]

            val_data = data[train_size:, :, :, :]
            val_labels = labels[train_size:, :]
            return (train_data, train_labels), (val_data, val_labels)

        else:

            return (data, labels)

    @staticmethod
    def ID(test):
        path = '../data/cifar-10-batches-py/'
        if not test:
            data = []
            labels = []
            for i in [1, 2, 3, 4, 5]:
                import pickle
                with open(path + 'data_batch_' + str(i), 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    # print(dict.keys())
                    data.append(dict[b'data'])
                    labels.append(dict[b'labels'])
            data = torch.from_numpy(np.concatenate(data)).float().reshape(
                50000, 3, 32, 32).transpose(2, 3)
            labels = torch.from_numpy(np.concatenate(labels)).unsqueeze(1)
            flag = torch.tensor([1]*labels.shape[0]).unsqueeze(1)
            labels = torch.cat((labels, flag), 1)

            idx = torch.randperm(data.shape[0])
            data = data[idx].view(data.size())
            labels = labels[idx].view(labels.size())

            train_size = int(0.90 * len(data))
            train_data = data[:train_size, :, :, :]
            train_labels = labels[:train_size, :]

            val_data = data[train_size:, :, :, :]
            val_labels = labels[train_size:, :]
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
            data = torch.from_numpy(np.concatenate(data)).float().reshape(
                10000, 3, 32, 32).transpose(2, 3)
            labels = torch.from_numpy(np.concatenate(labels)).unsqueeze(1)
            flag = torch.tensor([1]*labels.shape[0]).unsqueeze(1)
            labels = torch.cat((labels, flag), 1)
            return (data, labels)

    def __getitem__(self, index):
        return self.normalize(self.data[index]), self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


if __name__ == "__main__":

    # CIFAR10 Test===  #
    # # Download CIFAR10 dataset
    # torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
    # torchvision.datasets.CIFAR10(root='../data', train=False, download=True)

    # Download CIFAR10-C dataset

    train, val = cifar.read(test=False, only_id=True, data_aug=True, split_train_set=True,
                            train_proportion=0.8)  # DMNIST.read(test=True, data_aug=True)

    print(type(train), type(val))
    print(train.data.shape, val.data.shape)

    def test(data):
        dl = DataLoader(data, batch_size=1024, shuffle=True)
        for batch in dl:
            img, label = batch
            print(label)
            print(label.shape)
            print(img.shape)
            print("Batch mean ", img.mean(dim=[0, 2, 3]))
            print("Batch std ", img.std(dim=[0, 2, 3]))
            break
    test(train)
    test(val)
