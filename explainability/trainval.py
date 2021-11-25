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



def trainval(exp_dict, savedir_base, data_root, do_pretraining=False, reset=False):
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
    model = get_model(exp_dict, labelset=train_dataset.raw_labelset, writer=None)
    print("Model with:", sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")

    # Checkpoint
    # -----------
    model_path = os.path.join(savedir, "model.pth")
    pretrain_score_list_path = os.path.join(savedir, "pretrain_score_list.pkl")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(pretrain_score_list_path):
        # resume pretraining
        model.load_state_dict(hu.torch_load(model_path))
        pretrain_score_list = hu.load_pkl(pretrain_score_list_path)
        pretrain_epoch = pretrain_score_list[-1]["pretrain_epoch"] + 1

    elif exp_dict["encoder_weights"] is not None:
        model.load_encoder(hu.torch_load(exp_dict["encoder_weights"]))

        # continue decoder training
        if os.path.exists(score_list_path):
            score_list = hu.load_pkl(score_list_path)
            epoch = score_list[-1]["epoch"] + 1

        # start decoder training
        else:
            score_list = []
            epoch = 0

    else:
        # start pretraining
        pretrain_score_list = []
        pretrain_epoch = 0
        score_list = []
        epoch = 0

    # Train & Val
    # ------------
    if do_pretraining:
        print("Starting pretraining at epoch %d" % (pretrain_epoch))

        for e in range(pretrain_epoch, exp_dict['max_epoch']):

            score_dict = {}
            # Train the model
            score_dict.update(model.train_on_loader(e, train_loader))

            # Validate the model
            score_dict.update(model.val_on_loader(e, val_loader))
            score_dict["pretrain_epoch"] = e

            wandb.log(score_dict, step=e, commit=True)
            pretrain_score_list += [score_dict]
            report_and_save(pretrain_score_list, model, pretrain_score_list_path, model_path, savedir)

        model.reinitialize_optim()

        print("Pretraining finished")

    print("Starting training at epoch %d" % (epoch))

    for e in range(epoch, exp_dict["max_epoch"]):

        score_dict = {}

        # Train the model
        score_dict.update(model.train_on_loader(e, train_loader))

        # Validate the model
        score_dict.update(model.val_on_loader(e, val_loader))
        score_dict["epoch"] = e

        wandb.log(score_dict, step=e + exp_dict["max_epoch"], commit=True)
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
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--pretraining", action="store_true")
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
            if not args.dry_run:
                wandb.init(project=args.project, dir=args.savedir_base, config=exp_dict, reinit=True)
            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    data_root=args.data_root,
                    do_pretraining=args.pretraining,
                    reset=args.reset)

            wandb.finish()
