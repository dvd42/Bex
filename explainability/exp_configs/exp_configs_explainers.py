from haven import haven_utils as hu

solid = {
    "backend": "synbols_hdf5",
    "n_continuous": 6,
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "non-camou-bw_n=1000000_2021-Sep-27.h5py",
    "task": "char",
    "augmentation": False,
    "mask": "random",
    "return_attributes": True,
}

solid_small = {
    "backend": "synbols_hdf5",
    "n_continuous": 6,
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "non-camou-bw_n=100000_2021-Aug-31.h5py",
    "task": "char",
    "augmentation": False,
    "mask": "random",
    "return_attributes": True,
}

generated = {
    "backend": "generated_synbols",
    "n_continuous": 6,
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "dataset-generated-font-char.h5py",
    "num_classes": 2,
    "augmentation": False,
    "mask": "random",
    "return_attributes": True}


biggan_decoder = {
    "name": "biggan_decoder",
    "mlp_width": 2,
    "mlp_depth": 2,
    "channels_width": 4,
    "dp_prob": 0.3,
    "feature_extractor": "resnet"
}

biggan_encoder = {
    "name": "biggan_encoder",
    "mlp_width": 2,
    "mlp_depth": 2,
    "channels_width": 4,
    "dp_prob": 0.3,
    "feature_extractor": "resnet"
}


oracle_dict = {"lr": 0.001,
                "weight_decay": 1e-4,
                "ngpu": 1,
                "batch_size": 64,
                "seed": 123,
                "model": "oracle",
                "backbone": biggan_encoder,
                "weights": "pretrained_models/model/oracle2.pth",
                "z_dim": 128,
                "max_epoch": 200,
                "dataset": solid}

generator_dict = {"lr": 0.001,
                "weight_decay": 1e-4,
                "oracle_dict": oracle_dict,
                "weights": "pretrained_models/model/generator2.pth",
                "ngpu": 1,
                "batch_size": 64,
                "seed": 123,
                "lambda": 0.01,
                "model": "generator",
                "backbone": biggan_decoder,
                "z_dim": 128,
                "max_epoch": 200,
                "alpha": 0.20,
                "dataset": solid}


classifier_dict = {"ngpu": 1,
                   "weight_decay": 1e-4,
                   "seed": 123,
                   "lr": 0.01,
                   "batch_size": 256,
                   "backbone": {"name" :"resnet18"},
                   "weights": "pretrained_models/classifier/classifier_font_char.pth",
                   "model": "resnet",
                   "max_epoch": 2,
                   "dataset": generated}

EXP_GROUPS = {}
EXP_GROUPS["dive"] = hu.cartesian_exp_group({"generator_dict": generator_dict,
                        "classifier_dict": classifier_dict,
                        "seed": 42,
                        "explainer": "dive",
                        "lr": 0.01,
                        "max_iters": 20,
                        "cache_batch_size": 64,
                        "force_cache": False,
                        "batch_size": 100,
                        # if x% of the counterfactuals are successful
                        "stop_batch_threshold": 0.9,
                        "num_explanations": 8,
                        "method": "fisher_spectral_inv",
                        # "method": "else",
                        "reconstruction_weight": 1,
                        "lasso_weight": 1.,
                        "diversity_weight": 1,
                        "n_samples": 100,
                        "fisher_samples": 0,
                        "dataset": generated})

import numpy as np
from copy import deepcopy
np.random.seed(42)
random_search = EXP_GROUPS['dive'][0] # default hparams
n_trials = 5
lasso_space = [0] + list([1, 5, 10])
lr_space = [0.005, 0.01, 0.05, 0.1]
diversity_space = [0] + list([0.001, 0.01, 0.1, 1, 10])
reconstruction_space = [0] + list([1, 5, 10])
fisher_samples = list([10])
methods = ['fisher_spectral', 'random', 'dive', 'fisher_spectral_inv']
exps = []
n = 0
for run in range(n_trials):
    base_exp = deepcopy(random_search)
    base_exp['lr'] = float(np.random.choice(lr_space))
    base_exp['lasso_weight'] = float(np.random.choice(lasso_space))
    base_exp['diversity_weight'] = float(np.random.choice(diversity_space))
    base_exp['reconstruction_weight'] = float(
        np.random.choice(reconstruction_space))
    for method in methods:
        method_exp = deepcopy(base_exp)
        method_exp['method'] = method
        if method_exp in exps:
            continue
        exps.append(method_exp)
EXP_GROUPS['random_search'] = exps
