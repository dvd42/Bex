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
    'https://zenodo.org/record/4701316/files/%s?download=1',
    'https://github.com/ElementAI/synbols-resources/raw/master/datasets/generated/%s'
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
                print("Downloading part %d/%d" %(i + 1, len(parts)))
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

    if dataset_dict["name"] == "uniform_z":

        ret = []
        for split in splits:

            ratio = dataset_dict["ratios"][0] if split == "train" else dataset_dict["ratios"][1]
            data = ToyZ(n_attributes=dataset_dict["n_attributes"],
                        n_samples=int(dataset_dict["n_samples"] * ratio))

            ret.append(data)


    elif dataset_dict["backend"] == "generated_synbols":
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

    else:
        raise ValueError


    return ret


def _read_json_key(args):
    string, key = args
    return json.loads(string)[key]


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
            print("Converting json strings to labels...")
            with multiprocessing.Pool(8) as pool:
                self.y = pool.map(json.loads, y)
            print("Done converting.")

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

                print("Done parsing raw labels.")
            else:
                self.raw_labels = None

            print("Generating labels")
            self.y = self.generate_labels()
            print("Done reading hdf5.")



    def generate_labels(self):

        att = {}

        for item in self.raw_labels:
            for k, v in item.items():
                if k in att:
                    att[k].append(v)
                else:
                    att[k] = [v]

        df = pd.DataFrame.from_dict(att)

        # ones = (df["scale"] > 0.3) & (df["rotation"] < 0)
        ones = (df["char"] % 2 == 1)
        # ones = (df["translation-x"] > 0.5) & (df["rotation"] < 0)
        # ones = df["inverse_color"] > 0.5

        labels = np.zeros(len(self.y))
        labels[ones] = 1

        return labels


class SynbolsSplit(Dataset):
    def __init__(self, dataset, split, transform=None):
        self.path = dataset.path
        self.mask = dataset.mask
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

    def oracle(self, z, decoder):

        weights = decoder.char_embedding.weight
        weights = weights[None, ...]
        # first 128 are the embedding of char class
        z = z[:, None, :128]

        preds = torch.linalg.norm(weights - z, dim=-1).argmin(-1)
        ones = preds % 2 == 1
        oracle_labels = torch.zeros_like(preds)
        oracle_labels[ones] = 1


        return oracle_labels

    def __getitem__(self, item):
        if self.raw_labels is None or not self.return_attributes:
            return self.transform(self.x[item]), self.y[item]

        curr_labels = copy.deepcopy(self.raw_labels[item])
        if "seed" in curr_labels:
            curr_labels.pop("seed") # not useful
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


class ToyZ(Dataset):

    def __init__(self, n_attributes, n_samples):
        self.n_attributes = n_attributes
        self.n_samples = n_samples

        self.x = torch.zeros(n_samples, n_attributes)
        self.x.uniform_(0, 1)

        self.y = self.generate_labels()


    def oracle(self, z):

        ones = z[..., 1] < 0.5
        labels = torch.zeros_like(ones).long()
        labels[ones] = 1

        return labels


    def generate_labels(self):

        return self.oracle(self.x).numpy()


    def __len__(self):
        return len(self.y)


    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]






