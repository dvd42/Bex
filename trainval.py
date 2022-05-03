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
torch.backends.cudnn.benchmark = True


def report_and_save(score_list, model, score_list_path, model_path, savedir):

    score_df = pd.DataFrame(score_list)
    print("\n", score_df.tail())
    hu.torch_save(model_path, model.get_state_dict())
    hu.save_pkl(score_list_path, score_list)
    print("Checkpoint Saved: %s" % savedir)


def trainval(exp_dict, exp_group_name, savedir_base, data_root, reset=False):
    # bookkeeping
    # ---------------
    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    np.random.seed(exp_dict["seed"])
    torch.manual_seed(exp_dict["seed"])

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)

    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
    pprint.pprint(exp_dict)
    print("Experiment saved in %s" % savedir)

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
    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(score_list_path):
        score_list = hu.load_pkl(score_list_path)
        epoch = score_list[-1]["epoch"] + 1

    else:
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
        report_and_save(score_list, model, score_list_path, model_path, savedir)


    print('experiment completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--data_root', default="", type=str)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, "exp_dict.json"))

        exp_list = [exp_dict]

    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list.append(default_configs[exp_group_name])


    # run experiments
    for exp_group_name, exp_dict in zip(args.exp_group_list, exp_list):
        # do trainval
        trainval(exp_dict=exp_dict,
                 exp_group_name=exp_group_name,
                savedir_base=args.savedir_base,
                data_root=args.data_root,
                reset=args.reset)
