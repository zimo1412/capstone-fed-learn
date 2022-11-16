import os

import torch
from torchvision import datasets


COVID_ROOT = os.environ["COVID_ROOT"]

class COVID_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, data_idx = None, transform = None):
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
        self.transform = transform
        self.data, self.index_dict = self.__build_covid_subset__()

    def __build_covid_subset__(self):
        # if index provided, extract subset, otherwise use the whole set

        # ImageFolder's return can be taken as input to Dataloader directly
        covid_dataobj = datasets.ImageFolder(root = self.root, transform = self.transform)
        data = covid_dataobj
        if self.data_idx is not None:
            index_dict = {i: self.data_idx[i] for i in range(len(self.data_idx))}
        else:
            index_dict = {i: i for i in range(len(covid_dataobj))}
        return data, index_dict

    def __getitem__(self, index):
        return self.data[self.index_dict[index]]

    def __len__(self):
        return len(self.index_dict)
