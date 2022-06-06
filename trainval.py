import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import time
import os
import argparse

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc

from explainability_benchmark.datasets import get_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from explainability_benchmark.models.configs import default_configs
from explainability_benchmark.models import get_model
import pandas as pd
import pprint
import torch
import numpy as np


def report_and_save(score_list, model, model_path, savedir):

    score_df = pd.DataFrame(score_list)
    print("\n", score_df.tail())
    hu.torch_save(model_path, model.get_state_dict())
    print("Checkpoint Saved: %s" % savedir)


def trainval(exp_dict, exp_group_name, savedir, data_root, corr, n_clusters, reset=False):
    torch.backends.cudnn.benchmark = True
    # bookkeeping
    # ---------------
    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)

    np.random.seed(exp_dict["seed"])
    torch.manual_seed(exp_dict["seed"])

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)

    # create folder and save the experiment dictionary
    if "generator" in exp_dict:
        exp_dict["generator"]["weights"] = os.path.join(data_root, exp_dict["generator"]["weights"])
    os.makedirs(savedir, exist_ok=True)
    exp_dict["weights"] = None
    pprint.pprint(exp_dict)
    print("Model saved in %s" % savedir)

    # Dataset
    # -----------
    train_dataset, val_dataset = get_dataset(['train', 'val'], data_root, exp_dict)
    # val_dataset = get_dataset('val', exp_dict)

    # train and val loader
    train_loader = DataLoader(train_dataset,
                                batch_size=exp_dict['batch_size'],
                                shuffle=True,
                                num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset,
                                batch_size=exp_dict['batch_size'],
                                shuffle=False,
                                num_workers=args.num_workers)
    # Model
    # -----------
    model = get_model(exp_group_name, savedir)
    print("Model with:", sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")

    # Checkpoint
    # -----------

    model_path = os.path.join(savedir, f"{exp_group_name}_corr{corr}_n_clusters{n_clusters}.pth")
    score_list = []
    epoch = 0

    # Train & Val
    # ------------
    print("Starting training at epoch %d" % (epoch))

    for e in range(epoch, exp_dict["max_epoch"]):

        score_dict = {}

        # Train the model
        score_dict.update(model.train_on_loader(e, train_loader))

        # Validate the model
        score_dict.update(model.val_on_loader(e, val_loader))
        score_dict["epoch"] = e

        score_list += [score_dict]
        report_and_save(score_list, model, model_path, savedir)


    print('experiment completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir', required=True)
    parser.add_argument('-d', '--data_root', default="", type=str)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)
    parser.add_argument("--corr_level", type=float, default=0.95)
    parser.add_argument("--n_clusters_att", type=int, default=2)

    args = parser.parse_args()

    # Collect experiments
    # -------------------

    exp_list = []
    for exp_group_name in args.exp_group_list:
            exp_list.append(default_configs[exp_group_name])


    # run experiments
    for exp_group_name, exp_dict in zip(args.exp_group_list, exp_list):
        # do trainval
        exp_dict["dataset"]["name"] += f"_corr{args.corr_level}_n_clusters{args.n_clusters_att}.h5py"

        trainval(exp_dict=exp_dict, exp_group_name=exp_group_name, savedir=args.savedir,
                 data_root=args.data_root, corr=args.corr_level,
                 n_clusters=args.n_clusters_att, reset=args.reset)
