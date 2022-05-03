import os
import copy
import json
import h5py
import torch
from tqdm import tqdm
from haven import haven_utils as hu
from torch.utils.data import DataLoader
import numpy as np
from explainability_benchmark.datasets import get_dataset
from explainability_benchmark.models.configs import default_configs
from explainability_benchmark.models import get_model


noise_level = 0.05
idx_to_name = {0: "inverse_color", 1: "pixel_noise_scale", 2:"scale", 3: "translation-x", 4:"translation-y", 5: "rotation"}


def get_predictions(model, batch, mode):

    attributes = {}
    attr_dumps = []
    x, y, categorical_att, continuous_att = batch

    # char -2 font -1
    characters = torch.randint(47, size=(x.shape[0], ))
    categorical_att[:, -2] = characters

    y = characters % 2 == 1
    flip = torch.rand(x.shape[0])
    _y = torch.where(flip < noise_level, ~y, y)
    # font
    # categorical_att[:, -1].masked_fill_(_y, 42)
    # categorical_att[:, -1].masked_fill_(~_y, 200)
    # y = _y
    # # color
    # if mode == "training":
    # continuous_att[:, 0].masked_fill_(_y, 1)
    # continuous_att[:, 0].masked_fill_(~_y, 0)

    # translation
    right = torch.zeros(_y.shape[0]).uniform_(0, 1)
    left = torch.zeros_like(right).uniform_(-1, 0)
    continuous_att[:, 3] = torch.where(_y, right, left)

    reconstruction = model.predict_on_batch(categorical_att, continuous_att)

    pred_continuous = continuous_att
    pred_font = categorical_att[:, -1]
    pred_char = categorical_att[:, -2]

    for p_cont, p_font, p_char in zip(pred_continuous, pred_font, pred_char):
        for i in range(p_cont.size(0)):
            attributes[idx_to_name[i]] = p_cont[i].item()
        attributes["char"] = p_char.item()
        attributes["font"] = p_font.item()

        attr_dumps.append(json.dumps(attributes))

    images = reconstruction.cpu().numpy()

    return attr_dumps, images, y


@torch.no_grad()
def create_dataset(model, train_loader, val_loader, save_path):

    model.eval()
    f = h5py.File(save_path, "w")
    length = len(train_loader.dataset) + len(val_loader.dataset)
    shape = train_loader.dataset[0][0].size()
    f.create_dataset("x", (length, shape[1], shape[2], shape[0]), dtype="uint8")
    attributes = []


    last_write = 0
    labels = []
    for loader in [train_loader, val_loader]:
        if loader.dataset.x.shape[0] > 200000:
            mode = "training"
        else:
            mode = "val"
        print(mode)
        for i, batch in enumerate(tqdm(loader)):

            attr_str, images, y = get_predictions(model, batch, mode)

            images = np.array(images).transpose(0, 2, 3, 1)
            mean = np.array([0.5] * 3)
            images = images * mean + mean # denormalize
            images = (images * 255).astype("uint8")
            start = i * images.shape[0] + last_write
            end = start + images.shape[0]
            f["x"][start: end] = images
            attributes += attr_str
            labels += y.tolist()

        last_write = end
    f.create_dataset("y", data=attributes)
    f.create_dataset("labels", data=labels)
    f.close()




def generate_dataset(data_root, exp_dict):

    dataset_dict = copy.deepcopy(exp_dict)
    dataset_dict["dataset"]["mask"] = None
    train_dataset, val_dataset = get_dataset(['train', 'val'], data_root, dataset_dict)

    # train and val loader
    train_loader = DataLoader(train_dataset,
                                batch_size=exp_dict['batch_size'],
                                shuffle=False,
                                num_workers=4)
    val_loader = DataLoader(val_dataset,
                                batch_size=exp_dict['batch_size'],
                                shuffle=False,
                            num_workers=4)


    # Necessary to load oracle
    # exp_dict["generator_dict"]["savedir_base"] = args.savedir_base
    model = get_model("generator", data_root)
    weights = exp_dict["weights"]
    model.load_state_dict(hu.torch_load(weights))
    path = os.path.join(data_root, "dataset-translation-correlation.h5py")
    create_dataset(model, train_loader, val_loader, path)


if __name__ == "__main__":

    exp_dict = default_configs["generator"]
    generate_dataset("explainability_benchmark/data", exp_dict)

