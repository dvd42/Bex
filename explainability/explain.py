import torch.multiprocessing
import h5py
torch.multiprocessing.set_sharing_strategy("file_system")
import time
import os
import argparse
import wandb

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc

from torch.utils.data import DataLoader
import torchvision.transforms as tt
from exp_configs import EXP_GROUPS
from explainers import get_explainer
import pandas as pd
import pprint
import torch
import numpy as np
from torchvision.utils import make_grid
torch.backends.cudnn.benchmark = True


def report_and_save(score_list, model, score_list_path, model_path, savedir):

    score_df = pd.DataFrame(score_list)
    print("\n", score_df.tail())
    hu.torch_save(model_path, model.get_state_dict())
    hu.save_pkl(score_list_path, score_list)
    print("Checkpoint Saved: %s" % savedir)



def explain(exp_dict, savedir_base, data_root, reset):
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

    # Model
    # -----------
    # exp_dict["generator_dict"]["savedir_base"] = savedir_base
    explainer = get_explainer(exp_dict, savedir=savedir, data_path=data_root)


    print("Running explainer")
    attack_histories = explainer.attack_dataset()

    mean = torch.tensor([0.5] * 3)[None, None, :, None, None]
    images = np.concatenate(attack_histories["images"])[:, None, ...]
    decoded = np.concatenate(attack_histories["decoded"])
    to_log = torch.tensor(np.hstack((images, decoded)))
    to_log = to_log * mean + mean

    f, ax = plt.subplots(1, explainer.appended_images)
    f.set_size_inches(18.5, 10.5)
    for i, batch in enumerate(to_log.chunk(explainer.appended_images)):
        grid = make_grid(batch.view(-1, *decoded.shape[2:]), nrow=9).permute(1, 2, 0).numpy()
        ax[i].imshow(grid)
        ax[i].set_axis_off()

    wandb.log({"counterfactuals": f}, commit=True)

    print(f"Saving results to {savedir}")
    with h5py.File(os.path.join(savedir, 'results.h5'), 'w') as outfile:
        for k, v in attack_histories.items():
            print(f'saving {k}')
            try:
                outfile[k] = np.concatenate(v, 0)
            except ValueError:
                outfile[k] = v

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
            # run explainer
            wandb.init(project=args.project, dir=args.savedir_base, config=exp_dict, reinit=True)
            explain(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    data_root=args.data_root,
                    reset=args.reset)

            wandb.finish()
