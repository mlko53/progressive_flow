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
            self.data = datasets.MNIST(train=train, download=True, transform=transform)
            if return_tuple:
                self.downsampled_data = datasets.MNIST(train=train, download=True, transform=downsampled_transform)
        if name == "CIFAR":
            self.data = datasets.CIFAR10(train=train, download=True, transform=transform)
            if return_tuple:
                self.downsampled_data = datasets.CIFAR10(train=train, download=True, transform=downsampled_transform)
        if name == "CELEBA":
            assert(False)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.return_tuple:
            return (self.data[index], self.downsampled_data[index])
        else:
            return self.data[index]
    
    

