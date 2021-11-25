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
biggan_biggan = {
    "name": "biggan",
    "mlp_width": 2,
    "mlp_depth": 2,
    "channels_width": 4,
    "dp_prob": 0.3,
    "feature_extractor": "resnet"
}

EXP_GROUPS = {}
EXP_GROUPS["autoencoder"] = hu.cartesian_exp_group({'lr': [0.001],
                        'weight_decay': [1e-4],
                        'ngpu': 1,
                        'batch_size': [64],
                        'seed': [123],
                        'lambda': 0.001,
                        'model': "autoencoder",
                        'backbone': [biggan_biggan],
                        'z_dim': [128],
                        'max_epoch': 200,
                        'alpha': [0.10],
                        'dataset': [solid]})
