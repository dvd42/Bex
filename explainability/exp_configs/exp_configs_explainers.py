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
    "return_attributes": False,
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
    "return_attributes": False,
}
vae = {
    "name": "vae",
    "mlp_width": 2,
    "mlp_depth": 2,
    "channels_width": 4,
    "dp_prob": 0.3,
    "feature_extractor": "resnet"
}

EXP_GROUPS = {}
EXP_GROUPS["tcvae"] = hu.cartesian_exp_group({"lr": [0.001],
                        "weight_decay": [1e-4],
                        "encoder_weights": None,
                        "ngpu": 1,
                        "batch_size": [64],
                        "seed": [123],
                        "z_dim": 128,
                        "model": "tcvae",
                        "backbone": vae,
                        "beta": [0.001], # the idea is to be able to interpolate while getting good reconstructions
                        "tc_weight": [1], # we keep the total_correlation penalty high to encourage disentanglement
                        "vgg_weight": [1],
                        "beta_annealing": [True],
                        "max_epoch": 10,
                        "dataset": [solid_small]})
