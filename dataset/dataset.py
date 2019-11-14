from pathlib import Path
from PIL import Image

import torch
from torch.utils import data 
from torchvision import transforms, datasets

class Dataset(data.Dataset):

    def __init__(self, split, name, resolution, return_tuple):
        self.split = split
        self.name = name
        self.resolution = resolution
        self.return_tuple = return_tuple

        transform = transforms.Compose([
            transforms.Resize(resolution), 
            transforms.ToTensor()
        ])
        
        downsampled_transform = transforms.Compose([
            transforms.Resize(resolution // 2), 
            transforms.ToTensor()
        ])

        train = (split == "train")
        if name == "MNIST":
            self.data = datasets.MNIST(root="data/", train=train, download=True, transform=transform)
            if return_tuple:
                self.downsampled_data = datasets.MNIST(root="data/", train=train, download=True, transform=downsampled_transform)
        if name == "CIFAR":
            self.data = datasets.CIFAR10(root="data/", train=train, download=True, transform=transform)
            if return_tuple:
                self.downsampled_data = datasets.CIFAR10(root="data/", train=train, download=True, transform=downsampled_transform)
        if name == "CelebA":
            self.data = Celeba_Dataset(split=split, transform=transform)
            if return_tuple:
                self.downsampled_data = Celeba_dataset(split=split, transform=transform)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.return_tuple:
            return (self.data[index], self.downsampled_data[index])
        else:
            return self.data[index]

class Celeba_Dataset(data.Dataset):
    def __init__(self, split, transform):
        self.root = Path("/deep/u/mlko53/celeba")
        self.partition = pd.read_csv(self.root / "list_eval_partition.txt", header=None, sep=" ")
        self.partition.columns = ['file', 'split']
        if split == "train":
            self.data = self.partition[self.partition['split'] == 0]['file'].tolist()
        if split == "val":
            self.data = self.partition[self.partition['split'] == 1]['file'].tolist()
        if split == "test":
            self.data = self.partition[self.partition['split'] == 2]['file'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = Image.open(self.root / self.data[index])
        x = self.transform(x)
        return x
