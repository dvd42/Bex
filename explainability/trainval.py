import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import time
import os
import argparse
import wandb

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc

from datasets import get_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from exp_configs import EXP_GROUPS
from models import get_model
import pandas as pd
import pprint
import torch
import numpy as np
import tensorboardX
torch.backends.cudnn.benchmark = True


def report_and_save(score_list, model, score_list_path, model_path, savedir):

    score_df = pd.DataFrame(score_list)
    print("\n", score_df.tail())
    hu.torch_save(model_path, model.get_state_dict())
    hu.save_pkl(score_list_path, score_list)
    print("Checkpoint Saved: %s" % savedir)



def trainval(exp_dict, savedir_base, data_root, reset=False):
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

    # char_2_font = {}
    # char_2_img = {}
    # import matplotlib
    # # matplotlib.use("TkAgg")
    # import matplotlib.pyplot as plt
    # for i, x in enumerate(train_dataset.x):
    #     font = train_dataset.raw_labels[i]["font"]
    #     char = train_dataset.raw_labels[i]["char"]
    #     if char not in char_2_font:
    #         char_2_font[char] = []
    #         char_2_img[char] = []
    #     if font not in char_2_font[char]:
    #         char_2_font[char].append(font)

    #         char_2_img[char].append(x)
    #         if len(char_2_font[char]) >= 48:
    #             break
        # print(val_dataset.raw_labels[i])
        # print(val_dataset.raw_labels[i]["font"])
        # plt.imshow(x)
        # plt.show()


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
    exp_dict["savedir_base"] = savedir_base
    model = get_model(exp_dict, writer=None)
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

        wandb.log(score_dict, step=e, commit=True)
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
    parser.add_argument("--project", default="Synbols")

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
            exp_list += EXP_GROUPS[exp_group_name]


    # Run experiments or View them
    # ----------------------------
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")
    if args.run_jobs:
        from haven import haven_jobs as hjb
        from .job_config import job_config
        run_command = ('python trainval.py -ei <exp_id> -sb %s -nw %d -d %s' %  (args.savedir_base, args.num_workers, args.data_root))
        workdir = os.path.dirname(os.path.realpath(__file__))
        hjb.run_exp_list_jobs(exp_list,
                            savedir_base=args.savedir_base,
                            workdir=workdir,
                            run_command=run_command,
                            job_config=job_config)

    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            if not "dataset_name" in exp_dict:
                wandb.init(project=args.project, dir=args.savedir_base, config=exp_dict, reinit=True)
                trainval(exp_dict=exp_dict,
                        savedir_base=args.savedir_base,
                        data_root=args.data_root,
                        reset=args.reset)

                wandb.finish()
            else:
                from generate_dataset import generate_dataset
                generate_dataset(args, exp_dict)

