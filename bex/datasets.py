from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
import os
from torchvision import transforms as tt
from torchvision.datasets import MNIST, SVHN
from PIL import Image
import torch
import h5py
import multiprocessing
import sys
import copy
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import requests
from tqdm import tqdm

MIRRORS = [
    'https://zenodo.org/record/6616598/files/%s?download=1',
]

def get_data_path_or_download(dataset, data_root):
    """Finds a dataset locally and downloads if needed.

    Args:
        dataset (str): dataset name. For instance 'camouflage_n=100000_2020-Oct-19.h5py'.
            See https://github.com/ElementAI/synbols-resources/tree/master/datasets/generated for the complete list. (please ignore .a[a-z] extensions)
        data_root (str): path where the dataset will be or is stored. If empty string, it defaults to $TMPDIR

    Raises:
        ValueError: dataset name does not exist in local path nor in remote

    Returns:
        str: dataset final path
    """
    if data_root == "":
        data_root = os.environ.get("TMPDIR", "/tmp")
    full_path = os.path.join(data_root, dataset)

    if os.path.isfile(full_path):
        print("%s found." %full_path)
        return full_path
    else:
        print("Downloading %s..." %full_path)

    for i, mirror in enumerate(MIRRORS):
        try:
            download_url(mirror % dataset, full_path)
            return full_path
        except Exception as e:
            if i + 1 < len(MIRRORS):
                print("%s failed, trying %s..." %(mirror, MIRRORS[i+1]))
            else:
                raise e

def download_url(url, path):
    r = requests.head(url)
    if r.status_code == 302:
        raise RuntimeError("Server returned 302 status. \
            Try again later or contact us.")

    is_big = not r.ok
    if is_big:
        r = requests.head(url + ".aa")
        if not r.ok:
            raise ValueError("Dataset %s" %url, "not found in remote.")
        response = input("Download more than 3GB (Y/N)?: ").lower()
        while response not in ["y", "n"]:
            response = input("Download more than 3GB (Y/N)?: ").lower()
        if response == "n":
            print("Aborted")
            sys.exit(0)
        parts = []
        current_part = "a"
        while r.ok:
            r = requests.head(url + ".a%s" %current_part)
            parts.append(url + ".a" + current_part)
            current_part = chr(ord(current_part) + 1)
    else:
        parts = [url]

    if not os.path.isfile(path):
        with open(path, 'wb') as file:
            for i, part in enumerate(parts):
                # Streaming, so we can iterate over the response.
                response = requests.get(part, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024 #1 Kilobyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    file.close()
                    os.remove(path)
                    raise RuntimeError("ERROR, something went wrong downloading %s" %part)


def get_dataset(splits, data_root, exp_dict):

    if "dataset" in exp_dict:
        dataset_dict = exp_dict["dataset"]
    else:
        dataset_dict = exp_dict


    data_root = os.path.join(data_root, "datasets")
    os.makedirs(data_root, exist_ok=True)
    if dataset_dict["backend"] == "generated_synbols":
        full_path = get_data_path_or_download(dataset_dict["name"],
                                            data_root=data_root)

        data = GeneratedSynbols(full_path, dataset_dict["num_classes"],
                                return_attributes=dataset_dict["return_attributes"])

        ret = []

        for split in splits:
            transform = [tt.ToPILImage()]
            transform += [tt.ToTensor(),
                        tt.Normalize([0.5] * dataset_dict["channels"],
                                    [0.5] * dataset_dict["channels"])]

            transform = tt.Compose(transform)
            dataset = SynbolsSplit(data, split, transform=transform)
            ret.append(dataset)


    elif dataset_dict["backend"] == "synbols_hdf5":
        full_path = get_data_path_or_download(dataset_dict["name"],
                                            data_root=data_root)
        data = SynbolsHDF5(full_path,
                        dataset_dict["task"],
                        mask=dataset_dict["mask"],
                        trim_size=dataset_dict.get("trim_size", None),
                        return_attributes=dataset_dict["return_attributes"])
        ret = []
        for split in splits:
            transform = [tt.ToPILImage()]
            if dataset_dict["augmentation"] and split == "train":
                transform += [tt.RandomResizedCrop(size=(dataset_dict["height"], dataset_dict["width"]), scale=(0.8, 1)),
                            tt.RandomHorizontalFlip(),
                            tt.ColorJitter(0.4, 0.4, 0.4, 0.4)]
            transform += [tt.ToTensor(),
                        tt.Normalize([0.5] * dataset_dict["channels"],
                                    [0.5] * dataset_dict["channels"])]
            transform = tt.Compose(transform)
            dataset = SynbolsSplit(data, split, transform=transform)
            ret.append(dataset)

    else:
        raise ValueError


    return ret

class DatasetWrapper(torch.utils.data.Dataset):
    """Helper class to provide image id"""

    def __init__(self, dataset, indices=None):
        """Constructor
        Args:
        dataset (torch.utils.data.Dataset): Dataset object
        """
        self.dataset = dataset
        self.indices = indices
        if self.indices is None:
            self.indices = list(range(len(dataset)))

    def __getitem__(self, item):
        return (self.indices[item], *self.dataset[self.indices[item]])

    def __len__(self):
        return len(self.indices)


class SynbolsHDF5:
    def __init__(self, path, task, ratios=[0.8, 0.2], mask=None, trim_size=None, raw_labels=True, return_attributes=False):
        self.path = path
        self.task = task
        self.ratios = ratios
        self.return_attributes = return_attributes
        print("Loading hdf5...")
        with h5py.File(path, 'r') as data:
            self.x = data['x'][...]
            y = data['y'][...]
            print("Converting json strings to labels...")
            with multiprocessing.Pool(8) as pool:
                self.y = pool.map(json.loads, y)
            if isinstance(mask, str):
                if "split" in data:
                    if mask in data['split'] and mask == "random":
                        self.mask = data["split"][mask][...]
                    else:
                        self.mask = self.parse_mask(mask, ratios=ratios)
                else:
                    raise ValueError
            else:
                self.mask = mask

            if raw_labels:
                print("Parsing raw labels...")
                raw_labels = copy.deepcopy(self.y)
                self.raw_labels = []
                to_filter = ["resolution", "symbols", "background",
                             "foreground", "alphabet", "is_bold", "is_slant"]
                self.raw_labelset = {k: [] for k in raw_labels[0].keys()}
                for item in raw_labels:
                    ret = {}
                    for key in item.keys():
                        if key not in to_filter:
                            if key == "translation":
                                ret['translation-x'], \
                                    ret['translation-y'] = item[key]
                                self.raw_labelset['translation-x'] = []
                                self.raw_labelset['translation-y'] = []
                            elif not isinstance(item[key], float):
                                ret[key] = item[key]
                                self.raw_labelset[key].append(item[key])
                            else:
                                ret[key] = item[key]
                                self.raw_labelset[key] = []

                    self.raw_labels.append(ret)
                str2int = {}
                for k in self.raw_labelset.keys():
                    v = self.raw_labelset[k]
                    if len(v) > 0:
                        v = list(sorted(set(v)))
                        self.raw_labelset[k] = v
                        str2int[k] = {k: i for i, k in enumerate(v)}
                for item in self.raw_labels:
                    for k in str2int.keys():
                        item[k] = str2int[k][item[k]]

            else:
                self.raw_labels = None

            self.y = np.array([y[task] for y in self.y])
            self.trim_size = trim_size
            if trim_size is not None and len(self.x) > self.trim_size:
                self.mask = self.trim_dataset(self.mask)

    def trim_dataset(self, mask, train_size=60000, val_test_size=20000):
        labelset = np.sort(np.unique(self.y))
        counts = np.array([np.count_nonzero(self.y == y) for y in labelset])
        imxclass_train = int(np.ceil(train_size / len(labelset)))
        imxclass_val_test = int(np.ceil(val_test_size / len(labelset)))
        ind_train = np.arange(mask.shape[0])[mask[:, 0]]
        y_train = self.y[ind_train]
        ind_train = np.concatenate([np.random.permutation(ind_train[y_train == y])[
                                   :imxclass_train] for y in labelset], 0)
        ind_val = np.arange(mask.shape[0])[mask[:, 1]]
        y_val = self.y[ind_val]
        ind_val = np.concatenate([np.random.permutation(ind_val[y_val == y])[
                                 :imxclass_val_test] for y in labelset], 0)
        ind_test = np.arange(mask.shape[0])[mask[:, 2]]
        y_test = self.y[ind_test]
        ind_test = np.concatenate([np.random.permutation(ind_test[y_test == y])[
                                  :imxclass_val_test] for y in labelset], 0)
        current_mask = np.zeros_like(mask)
        current_mask[ind_train, 0] = True
        current_mask[ind_val, 1] = True
        current_mask[ind_test, 2] = True
        return current_mask

    def parse_mask(self, mask, ratios):
        args = mask.split("_")[1:]
        if "stratified" in mask:
            mask = 1
            for arg in args:
                if arg == 'translation-x':
                    def fn(x): return x['translation'][0]
                elif arg == 'translation-y':
                    def fn(x): return x['translation'][1]
                else:
                    def fn(x): return x[arg]
                mask *= get_stratified(self.y, fn,
                                       ratios=[ratios[1], ratios[0], ratios[2]])
            mask = mask[:, [1, 0, 2]]
        elif "compositional" in mask:
            partition_map = None
            if len(args) != 2:
                raise RuntimeError(
                    "Compositional splits must contain two fields to compose")
            for arg in args:
                if arg == 'translation-x':
                    def fn(x): return x['translation'][0]
                elif arg == 'translation-y':
                    def fn(x): return x['translation'][1]
                else:
                    def fn(x): return x[arg]
                if partition_map is None:
                    partition_map = get_stratified(self.y, fn, tomap=False)
                else:
                    _partition_map = get_stratified(self.y, fn, tomap=False)
                    partition_map = stratified_splits.compositional_split(
                        _partition_map, partition_map)
            partition_map = partition_map.astype(bool)
            mask = np.zeros_like(partition_map)
            for i, split in enumerate(np.argsort(partition_map.astype(int).sum(0))[::-1]):
                mask[:, i] = partition_map[:, split]
        else:
            raise ValueError
        return mask


class GeneratedSynbols:

    def __init__(self, path, num_classes, ratios=[0.8, 0.2], raw_labels=True, return_attributes=False):

        self.path = path
        self.num_classes = num_classes
        self.mask = None
        self.ratios = ratios
        self.raw_labels = raw_labels
        self.return_attributes = return_attributes

        print("Loading hdf5...")
        with h5py.File(path, 'r') as data:
            self.x = data['x'][...]
            y = data['y'][...]
            self.correlated = data["correlated_att"][...]
            print("Converting json strings to labels...")
            with multiprocessing.Pool(8) as pool:
                self.y = pool.map(json.loads, y)

            if raw_labels:
                print("Parsing raw labels...")
                raw_labels = copy.deepcopy(self.y)
                self.raw_labels = []
                self.raw_labelset = {k: [] for k in raw_labels[0]}
                for item in raw_labels:
                    ret = {}
                    for key in item:
                        if not isinstance(item[key], float):
                            ret[key] = item[key]
                            self.raw_labelset[key].append(item[key])
                        else:
                            ret[key] = item[key]
                            self.raw_labelset[key] = []

                    self.raw_labels.append(ret)
                # str2int = {}
                # for k in self.raw_labelset.keys():
                #     v = self.raw_labelset[k]
                #     if len(v) > 0:
                #         v = list(sorted(set(v)))
                #         self.raw_labelset[k] = v
                #         str2int[k] = {k: i for i, k in enumerate(v)}
                # for item in self.raw_labels:
                #     for k in str2int:
                #         item[k] = str2int[k][item[k]]

            else:
                self.raw_labels = None

            # self._y = self.generate_labels()
            self.y = data["labels"][...]


    def generate_labels(self):

        att = {}

        for item in self.raw_labels:
            for k, v in item.items():
                if k in att:
                    att[k].append(v)
                else:
                    att[k] = [v]

        df = pd.DataFrame.from_dict(att)

        ones = (df["char"] % 2 == 1)

        labels = np.zeros(len(self.y))
        labels[ones] = 1

        return labels


class SynbolsSplit(Dataset):
    def __init__(self, dataset, split, transform=None):
        self.path = dataset.path
        self.mask = dataset.mask
        self.dataset = dataset
        self.return_attributes = dataset.return_attributes
        self.raw_labelset = dataset.raw_labelset
        self.raw_labels = dataset.raw_labels
        self.ratios = dataset.ratios
        self.split = split
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        # self.split_data(dataset.x, dataset.y, None, [0.8, 0.2])
        self.split_data(dataset.x, dataset.y, dataset.mask, dataset.ratios)

    def split_data(self, x, y, mask, ratios):
        if mask is None:
            if self.split == 'train':
                start = 0
                end = int(ratios[0] * len(x))
            elif self.split == 'val':
                start = int(ratios[0] * len(x))
                end = int((ratios[0] + ratios[1]) * len(x))
            elif self.split == 'test':
                start = int((ratios[0] + ratios[1]) * len(x))
                end = len(x)
            rng=np.random.RandomState(42)
            indices = rng.permutation(len(x))
            indices = indices[start:end]
        else:
            mask = mask[:, ["train", "val", "test"].index(self.split)]
            indices = np.arange(len(y))  # 0....nsamples
            indices = indices[mask]
        self.labelset = list(sorted(set(y)))
        self.y = np.array([self.labelset.index(_y) for _y in y])
        self.x = x[indices]
        self.y = self.y[indices]
        if self.raw_labels is not None:
            self.raw_labels = np.array(self.raw_labels)[indices]

    def __getitem__(self, item):
        if self.raw_labels is None or not self.return_attributes:
            return self.transform(self.x[item]), self.y[item]

        curr_labels = copy.deepcopy(self.raw_labels[item])
        if "seed" in curr_labels:
            curr_labels.pop("seed") # not useful
        if "pixel_noise_scale" in curr_labels:
            curr_labels.pop("pixel_noise_scale")

        continuous_att = [curr_labels["inverse_color"]]
        curr_labels.pop("inverse_color")
        categorical_att = []

        for name, att in curr_labels.items():
            if len(self.raw_labelset[name]) > 1:
                categorical_att.append(att)

            else:
                continuous_att.append(att)

        return self.transform(self.x[item]), self.y[item], torch.tensor(categorical_att), torch.tensor(continuous_att)

    def __len__(self):
        return len(self.x)
