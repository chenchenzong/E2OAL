import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import clip
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import Counter

from torch.utils.data import SubsetRandomSampler
import pickle
import os
import time

from PIL import Image


def get_CLIP():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device, jit=False)  # RN50, RN101, RN50x4, ViT-B/32
    return model, preprocess


class CustomCIFAR10Dataset_train(Dataset):
    cifar100_dataset = None
    targets = None

    @classmethod
    def load_dataset(cls, root="./data/cifar10", train=True, download=True, transform=None):
        cls.cifar10_dataset = datasets.CIFAR10(root, train=train, download=download, transform=transform)
        cls.targets = cls.cifar10_dataset.targets

    def __init__(self):
        if CustomCIFAR10Dataset_train.cifar10_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

    def __getitem__(self, index):
        data_point, label = CustomCIFAR10Dataset_train.cifar10_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(CustomCIFAR10Dataset_train.cifar10_dataset)


class CustomCIFAR100Dataset_train(Dataset):
    cifar100_dataset = None
    targets = None

    @classmethod
    def load_dataset(cls, root="./data/cifar100", train=True, download=True, transform=None):
        cls.cifar100_dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)
        cls.targets = cls.cifar100_dataset.targets

    def __init__(self):
        if CustomCIFAR100Dataset_train.cifar100_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

    def __getitem__(self, index):
        data_point, label = CustomCIFAR100Dataset_train.cifar100_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(CustomCIFAR100Dataset_train.cifar100_dataset)


class Tiny_ImageNet(Dataset):
    def __init__(self, transform):

        labels = []
        image_names = []
        with open('./tiny-imagenet-200/wnids.txt') as wnid:
            for line in wnid:
                labels.append(line.strip('\n'))
        for label in labels:
            txt_path = './tiny-imagenet-200/train/'+label+'/'+label+'_boxes.txt'
            image_name = []
            with open(txt_path) as txt:
                for line in txt:
                    image_name.append(line.strip('\n').split('\t')[0])
            image_names.append(image_name)
        print(len(image_names[0]), len(labels))

        i = 0
        self.data = []
        for label in labels:
            image = []
            for image_name in image_names[i]:
                image_path = os.path.join('./tiny-imagenet-200/train', label, 'images', image_name)
                data = Image.open(image_path).convert("RGB")
                image.append(np.asarray(data))
                data.close()
            self.data.append(image)
            i = i + 1

        self.data = np.array(self.data)
        self.data = self.data.reshape(-1, 64, 64, 3)
        self.uq_idxs = range(self.data.shape[0])
        self.targets = [i//500 for i in self.uq_idxs]
        self.uq_idxs = np.asarray(self.uq_idxs)

        self.ToPILImage = transforms.ToPILImage()
        self.transform = transform

    def __getitem__(self, index):

        label = self.targets[index]
        image = self.data[index]
        image = self.ToPILImage(np.uint8(image))

        return index, (self.transform(image), label)

    def __len__(self):
        len = self.data.shape[0]
        return len


def CIFAR100_EXTRACT_ALL(pre_type, dataset, model, preprocess):
    model.eval()

    #################################################

    if pre_type == "CLIP":
        preprocess_rand = transforms.Compose([preprocess])

    #################################################

    if dataset == "cifar10":
        CustomCIFAR10Dataset_train.load_dataset(transform=preprocess_rand)
        train_data = CustomCIFAR10Dataset_train()
    else:
        CustomCIFAR100Dataset_train.load_dataset(transform=preprocess_rand)
        train_data = CustomCIFAR100Dataset_train()

    batch_size = 256
    print('Data Loader')
    data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    features = []
    all_labels = []
    sel_idx = []
    for batch_idx, (index, (data, labels)) in enumerate(data_loader):
        print(batch_idx)
        data = data.to(device)
        labels = labels.to(device)

        with torch.no_grad():

            if pre_type == "CLIP":
                extracted_feature = model.encode_image(data)

            features.append(extracted_feature)

            sel_idx.append(index)
            all_labels.append(labels)

    final_feat = torch.concat(features).to(device)
    sel_idx = torch.concat(sel_idx).to(device)
    labels = torch.concat(all_labels).to(device)

    ###########################################################

    feature_sel_idx = sel_idx.unsqueeze(1).repeat(1, final_feat.size()[1]).to(device)

    ordered_feature = torch.gather(final_feat, 0, feature_sel_idx)

    ###########################################################

    index_to_label = {}
    for idx, index in enumerate(sel_idx):
        index_to_label[index] = labels[idx]

    save_folder = "./features/" + pre_type

    isExist = os.path.exists(save_folder)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(save_folder)

    if dataset == "cifar10":

        torch.save(ordered_feature, save_folder + '/cifar10_features.pt')

    else:

        torch.save(ordered_feature, save_folder + '/cifar100_features.pt')


def ImageNet_EXTRACT_ALL(pre_type, model, preprocess, dataset):
    model.eval()
    start_time = time.time()

    if pre_type == "CLIP":
        preprocess_rand = transforms.Compose([preprocess])

    if dataset == "tinyimagenet":
        train_data = Tiny_ImageNet(transform=preprocess_rand)

    batch_size = 1024
    print('Data Loader')
    data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    features = []

    all_labels = []

    sel_idx = []
    for batch_idx, (index, (data, labels)) in enumerate(data_loader):
        print(batch_idx, labels, len(labels))
        data = data.to(device)
        labels = labels.to(device)

        with torch.no_grad():

            if pre_type == "CLIP":

                extracted_feature = model.encode_image(data)

            features.append(extracted_feature)

            sel_idx.append(index)
            all_labels.append(labels)

    final_feat = torch.cat(features, dim=0).to(device)
    sel_idx = torch.cat(sel_idx, dim=0).to(device)
    labels = torch.cat(all_labels, dim=0).to(device)

    ###########################################################

    feature_sel_idx = sel_idx.unsqueeze(1).repeat(1, final_feat.size()[1]).to(device)

    ordered_feature = torch.gather(final_feat, 0, feature_sel_idx)
    print(ordered_feature.shape)

    ###########################################################

    index_to_label = {}
    for idx, index in enumerate(sel_idx):
        index_to_label[index] = labels[idx]

    save_folder = "./features/" + "CLIP"

    isExist = os.path.exists(save_folder)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(save_folder)
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f'extract_time: {epoch_duration}')
    torch.save(ordered_feature, save_folder + '/tinyimagenet_features.pt')


def extracted_feature(pre_type, dataset):
    model, preprocess = get_CLIP()

    if dataset == "tinyimagenet":

        ImageNet_EXTRACT_ALL(pre_type, model, preprocess, dataset)

    else:

        CIFAR100_EXTRACT_ALL(pre_type, dataset, model, preprocess)


if __name__ == "__main__":

    for dataset in ["cifar10", "cifar100", "tinyimagenet"]:
        print(dataset)
        extracted_feature("CLIP", dataset)
