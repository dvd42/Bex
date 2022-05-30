from haven import haven_utils as hu

solid = {
    "backend": "synbols_hdf5",
    "n_continuous": 5,
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
    "n_continuous": 5,
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "non-camou-bw_n=200000_2022-May-30.h5py",
    "task": "char",
    "augmentation": False,
    "mask": "random",
    "return_attributes": True,
}

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
                "z_dim": 3,
                "max_epoch": 100,
                "dataset": solid_small}

EXP_GROUPS = {}
# Train generator
EXP_GROUPS["generator"] = hu.cartesian_exp_group({"lr": [0.001],
                        "weight_decay": [1e-4],
                        "oracle_dict": oracle_dict,
                        "ngpu": 1,
                        "batch_size": [64],
                        "seed": [123],
                        "lambda": 0.01,
                        "model": "generator",
                        "backbone": [biggan_decoder],
                        "z_dim": [3],
                        "max_epoch": 100,
                        "alpha": [0.20],
                        "dataset": [solid_small]})
