import os

import torch
import numpy as np
from torchvision import datasets


COVID_ROOT = os.environ["COVID_ROOT"]

class COVID_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, is_train = True, data_idx = None, transform = None,
                 train_size = 900, valid_size = 100):
        """COVID dataset with index to extract subset

        Args:
            root: data root (must follow ImageFolder structure)
                  See [Link](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)
            data_idx: to specify the data for a particular client site.
                If index provided, extract subset, otherwise use the whole set
            transform: image transforms
        Returns:
            A PyTorch dataset
        """
        self.root = root
        self.data_idx = data_idx
        self.is_train = is_train

        assert train_size + valid_size <= 1000
        self.train_size = train_size
        self.valid_size = valid_size

        self.transform = transform
        self.data, self.index_dict = self.__build_covid_subset__()

    def __build_covid_subset__(self):
        # if index provided, extract subset, otherwise use the whole set

        # ImageFolder's return can be taken as input to Dataloader directly
        covid_dataobj = datasets.ImageFolder(root = self.root, transform = self.transform)
        data = covid_dataobj

        if self.is_train:
            data.samples = covid_dataobj.samples[0: 0 + self.train_size] + \
                        covid_dataobj.samples[1000: 1000 + self.train_size] + \
                        covid_dataobj.samples[2000: 2000 + self.train_size] + \
                        covid_dataobj.samples[3000: 3000 + self.train_size]
            
            data.targets = [s[1] for s in data.samples]
        else:
            data.samples = covid_dataobj.samples[1000 - self.valid_size: 1000] + \
                        covid_dataobj.samples[2000 - self.valid_size: 2000] + \
                        covid_dataobj.samples[3000 - self.valid_size: 3000] + \
                        covid_dataobj.samples[4000 - self.valid_size: 4000]
            
            data.targets = [s[1] for s in data.samples]

        if self.data_idx is not None:
            index_dict = {i: self.data_idx[i] for i in range(len(self.data_idx))}
        else:
            index_dict = {i: i for i in range(len(data))}
        return data, index_dict

    def __getitem__(self, index):
        return self.data[self.index_dict[index]]

    def __len__(self):
        return len(self.index_dict)
