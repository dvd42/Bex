import os
import copy
import json
import h5py
import torch
from tqdm import tqdm
from haven import haven_utils as hu
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import numpy as np
from WhyY.datasets import get_dataset
from WhyY.models.configs import default_configs
from WhyY.models import get_model
import argparse


idx_to_name = {0: "inverse_color", 1: "scale", 2: "translation-x", 3:"translation-y", 4: "rotation"}
# torch.manual_seed(0)
# np.random.seed(0)



def parse_args():

    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("att", choices=["font", "rotation"])
    parser.add_argument("--corr_level", type=float, default=0.95)
    parser.add_argument("--p_corr", type=float, default=0.5)
    parser.add_argument("--n_clusters", type=int, default=2)

    return parser.parse_args()



def correlation_font(categorical_att, args, kmeans, weights):

    # char -2 font -1
    b = categorical_att.shape[0]
    ret = torch.zeros(b).long()
    n_fonts = 48
    characters = torch.randint(47, size=(b, ))
    categorical_att[:, -2] = characters
    _y = characters % 2 == 1
    ind = torch.arange(n_fonts)

    centers = torch.from_numpy(kmeans.cluster_centers_)
    corr_fonts = torch.linalg.norm(centers[:, None, :] - torch.from_numpy(weights)[None, ...], ord=2, dim=-1).argmin(-1)

    n_corr_fonts = int(centers.shape[0] // 2)
    f1 = corr_fonts[:n_corr_fonts]
    f2 = corr_fonts[n_corr_fonts:]

    flip = torch.rand(b)

    # 5% noise
    y = torch.where(torch.rand(b) < 0.05, ~_y, _y)
    mask = flip < args.corr_level
    y_1 = (y == 0) & (mask)
    y_2 = (y == 1) & (mask)

    ret[y_1] = torch.from_numpy(np.random.choice(f1, y_1.sum().item()))
    ret[y_2] = torch.from_numpy(np.random.choice(f2, y_2.sum().item()))
    ret[~mask] = torch.from_numpy(np.random.choice(ind, (~mask).sum().item()))

    categorical_att[:, -1] = ret

    return categorical_att, y, corr_fonts


def correlation_scale_rotation(categorical_att, continuous_att):

    b = continuous_att.shape[0]
    characters = torch.randint(47, size=(b, ))
    categorical_att[:, -2] = characters
    _y = characters % 2 == 1

    # 10% noise
    y = torch.where(torch.rand(b) < 0.10, ~_y, _y)

    small = torch.zeros(y.shape[0]).uniform_(0.44, 0.78)
    large = torch.zeros_like(small).uniform_(0.78, 1.10)

    rotation_space = np.linspace(-1.40, 1.40, 16)

    left = torch.zeros(y.shape[0]).uniform_(-1.40, 0)
    right = torch.zeros_like(small).uniform_(0, 1.40)

    continuous_att[:, 2] = torch.where(y, small, large)
    continuous_att[:, 5] = torch.where(y, left, right)

    return continuous_att, categorical_att


def get_predictions(model, batch, args, clusters, weights):

    attributes = {}
    attr_dumps = []
    x, y, categorical_att, continuous_att = batch

    if args.att == "font":
        categorical_att, y, corr_font = correlation_font(categorical_att, args, clusters, weights)

    else:
    # scale + rotation
        continuous_att, categorical_att = correlation_scale_rotation(categorical_att, continuous_att)

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

    return attr_dumps, images, y, corr_font


@torch.no_grad()
def create_dataset(model, train_loader, val_loader, save_path, args):

    model.eval()
    f = h5py.File(save_path, "w")
    length = len(train_loader.dataset) + len(val_loader.dataset)
    shape = train_loader.dataset[0][0].size()
    f.create_dataset("x", (length, shape[1], shape[2], shape[0]), dtype="uint8")
    attributes = []

    weight = model.model.font_embedding.weight.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(weight)


    last_write = 0
    labels = []
    for loader in [train_loader, val_loader]:
        for i, batch in enumerate(tqdm(loader)):

            attr_str, images, y, correlated_att = get_predictions(model, batch, args, kmeans, weight)

            if "correlated_att" not in f:
                f["correlated_att"] = correlated_att

            images = np.array(images).transpose(0, 2, 3, 1)
            mean = np.array([0.5] * 3)
            images = images * mean + mean # denormalize
            images = (images * 255).astype("uint8")
            # import matplotlib
            # matplotlib.use("TkAgg")
            # import matplotlib.pyplot as plt
            # from torchvision.utils import make_grid
            # fig, axs = plt.subplots(10, 6, figsize=(15, 6))
            # fig.subplots_adjust(hspace = .001, wspace=0)
            # axs = axs.ravel()
            # grid = make_grid(batch[0]).permute(1, 2, 0).numpy()
            # plt.imshow(grid)
            # plt.gca().set_axis_off()
            # plt.show()
            # for i, x in enumerate(batch[0][:60]):
            #     axs[i].set_yticks([])
            #     axs[i].set_xticks([])
            #     axs[i].imshow(np.array(x).transpose(1, 2, 0))
            # plt.show()
            start = i * images.shape[0] + last_write
            end = start + images.shape[0]
            f["x"][start: end] = images
            attributes += attr_str
            labels += y.tolist()

        last_write = end
    f.create_dataset("y", data=attributes)
    f.create_dataset("labels", data=labels)
    f.close()




def generate_dataset(data_root, exp_dict, args):

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

    # font_2_char = {}
    # font_2_img = {}
    # import matplotlib
    # matplotlib.use("TkAgg")
    # import matplotlib.pyplot as plt
    # for i, x in enumerate(train_dataset.x):
    #     font = train_dataset.raw_labels[i]["font"]
    #     char = train_dataset.raw_labels[i]["char"]
    #     if font not in font_2_char:
    #         font_2_char[font] = []
    #         font_2_img[font] = []
    #     if char not in font_2_char[font]:
    #         font_2_char[font].append(char)

    #         font_2_img[font].append(x)
            # if len(char_2_font[char]) >= 48:
            #     break
        # print(val_dataset.raw_labels[i])
        # print(val_dataset.raw_labels[i]["font"])
        # plt.imshow(x)
        # plt.show()

    # for font in font_2_img.keys():
    #     idxs = np.argsort(font_2_char[font])
    #     font_2_img[font] = np.stack(font_2_img[font])[idxs].tolist()
    # from torchvision.utils import make_grid
    # f, axis = plt.subplots(6, 8)
    # f.set_size_inches(15.5, 15.5)
    # for ax, font in zip(axis.ravel(), sorted(font_2_img.keys())):
    #     images = torch.from_numpy(np.stack(font_2_img[font])).permute(0, 3, 1, 2)
    #     ax.imshow(make_grid(images).permute(1, 2, 0))
    #     ax.axis('off')
    #     ax.set_title(font)
    #     ax.axis('off')

    # plt.show()
    model = get_model("generator", data_root)
    if args.att == "font":
        path = os.path.join(data_root, f"datasets/dataset_{args.att}_corr{args.corr_level}_n_clusters{args.n_clusters}.h5py")
        print(path)
    else:
        path = os.path.join(data_root, f"datasets/dataset_{args.att}_scale.h5py")
    create_dataset(model, train_loader, val_loader, path, args)


if __name__ == "__main__":

    args = parse_args()
    exp_dict = default_configs["generator"]
    generate_dataset("data", exp_dict, args)
