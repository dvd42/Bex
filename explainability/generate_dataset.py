import os
import copy
import argparse
from tqdm import tqdm
import torch
from haven import haven_utils as hu
from datasets import get_dataset
from torch.utils.data import DataLoader
from exp_configs import EXP_GROUPS
from models import get_model
import h5py
import numpy as np
import json
import time


idx_to_name = {0: "inverse_color", 1: "pixel_noise_scale", 2:"scale", 3: "translation-x", 4:"translation-y", 5: "rotation"}


def get_predictions(model, batch):

    attributes = {}
    attr_dumps = []
    x, y, categorical_att, continuous_att = batch

    outputs = {}
    # char -2 font -1
    mask = categorical_att[:, -2] % 2 == 1
    categorical_att[:, -1].masked_fill_(mask, 42)
    categorical_att[:, -1].masked_fill_(~mask, 200)
    reconstruction = model.predict_on_batch(categorical_att, continuous_att)
    outputs["reconstruction"] = model.oracle(reconstruction)

    pred_continuous = outputs["reconstruction"]["pred_continuous"]
    pred_font = outputs["reconstruction"]["pred_font"]
    pred_char = outputs["reconstruction"]["pred_char"]

    for p_cont, p_font, p_char in zip(pred_continuous, pred_font, pred_char):
        for i in range(p_cont.size(0)):
            attributes[idx_to_name[i]] = p_cont[i].item()
        attributes["char"] = p_char.argmax().item()
        attributes["font"] = p_font.argmax().item()

        attr_dumps.append(json.dumps(attributes))

    images = reconstruction.cpu().numpy()

    return attr_dumps, images


@torch.no_grad()
def create_dataset(model, train_loader, val_loader, save_path):

    model.eval()
    f = h5py.File(save_path, "w")
    length = len(train_loader.dataset) + len(val_loader.dataset)
    shape = train_loader.dataset[0][0].size()
    f.create_dataset("x", (length, shape[1], shape[2], shape[0]), dtype="uint8")
    attributes = []


    last_write = 0
    for loader in [train_loader, val_loader]:
        for i, batch in enumerate(tqdm(loader)):

            attr_str, images = get_predictions(model, batch)

            images = np.array(images).transpose(0, 2, 3, 1)
            mean = np.array([0.5] * 3)
            images = images * mean + mean # denormalize
            images = (images * 255).astype("uint8")
            start = i * images.shape[0] + last_write
            end = start + images.shape[0]
            f["x"][start: end] = images
            attributes += attr_str

        last_write = end
    f.create_dataset("y", data=attributes)
    f.close()




def generate_dataset(args, exp_dict):

    dataset_dict = copy.deepcopy(exp_dict)
    dataset_dict["dataset"]["mask"] = None
    train_dataset, val_dataset = get_dataset(['train', 'val'], args.data_root, dataset_dict)

    # train and val loader
    train_loader = DataLoader(train_dataset,
                                batch_size=exp_dict['batch_size'],
                                shuffle=False,
                                num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset,
                                batch_size=exp_dict['batch_size'],
                                shuffle=False,
                            num_workers=args.num_workers)


    # Necessary to load oracle
    exp_dict["generator_dict"]["savedir_base"] = args.savedir_base
    model = get_model(exp_dict["generator_dict"], labelset=train_dataset.raw_labelset, writer=None)
    if "weights" in exp_dict["generator_dict"]:
        weights = exp_dict["generator_dict"]["weights"]
    else:
        exp_dict["generator_dict"].pop("savedir_base")
        generator_dict = exp_dict["generator_dict"]
        weights = os.path.join(args.savedir_base, hu.hash_dict(generator_dict), "model.pth")

    model.load_state_dict(hu.torch_load(weights))
    path = os.path.join(args.data_root, exp_dict["dataset_name"])
    create_dataset(model, train_loader, val_loader, path)
    # necessary for h5py to close release the file
    time.sleep(5)
