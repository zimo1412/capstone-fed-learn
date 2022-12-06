import json
import os

import numpy as np
import torchvision.datasets as datasets
from pt.utils.covid_dataset import COVID_Dataset

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext

COVID_ROOT = os.environ["COVID_ROOT"]

def _get_site_class_summary(train_label, site_idx):
    class_sum = {}

    for site, data_idx in site_idx.items():
        unq, unq_cnt = np.unique(train_label[data_idx], return_counts=True)
        tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
        class_sum[site] = tmp
    return class_sum


class COVIDDataSplitter(FLComponent):
    def __init__(self, split_dir: str = None, num_sites: int = 8, alpha: float = 0.5, seed: int = 0):
        super().__init__()
        self.split_dir = split_dir
        self.num_sites = num_sites
        self.alpha = alpha
        self.seed = seed

        if alpha < 0.0:
            raise ValueError(f"Alpha should be larger 0.0 but was {alpha}!")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.split(fl_ctx)

    def split(self, fl_ctx: FLContext):
        np.random.seed(self.seed)

        self.log_info(
            fl_ctx,
            f"Partition COVID dataset into {self.num_sites} sites with Dirichlet sampling under alpha {self.alpha}",
        )
        site_idx, class_sum = self._partition_data()

        # write to files
        if self.split_dir is None:
            raise ValueError("You need to define a valid `split_dir` when splitting the data.")
        if not os.path.isdir(self.split_dir):
            os.makedirs(self.split_dir)
        sum_file_name = os.path.join(self.split_dir, "summary.txt")
        with open(sum_file_name, "w") as sum_file:
            sum_file.write(f"Number of clients: {self.num_sites} \n")
            sum_file.write(f"Dirichlet sampling parameter: {self.alpha} \n")
            sum_file.write("Class counts for each client: \n")
            sum_file.write(json.dumps(class_sum))

        site_file_path = os.path.join(self.split_dir, "site-")
        for site in range(self.num_sites):
            site_file_name = site_file_path + str(site + 1) + ".npy"
            np.save(site_file_name, np.array(site_idx[site]))

    def load_COVID_data(self):
        # load training data
        train_dataset = COVID_Dataset(root = COVID_ROOT)

        # only training label is needed for doing split
        train_label = np.array([x[1] for x in train_dataset.data])
        return train_label

    def _partition_data(self):
        train_label = self.load_COVID_data()

        min_size = 0
        K = 4
        N = train_label.shape[0]
        site_idx = {}

        # split
        while min_size < 10:
            idx_batch = [[] for _ in range(self.num_sites)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(train_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_sites))
                # Balance
                proportions = np.array(
                    [p * (len(idx_j) < N / self.num_sites) for p, idx_j in zip(proportions, idx_batch)]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        # shuffle
        for j in range(self.num_sites):
            np.random.shuffle(idx_batch[j])
            site_idx[j] = idx_batch[j]

        # collect class summary
        class_sum = _get_site_class_summary(train_label, site_idx)

        return site_idx, class_sum
