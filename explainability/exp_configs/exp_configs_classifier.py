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
                "dataset": solid_small}

generator_dict = {"lr": 0.001,
                "weight_decay": 1e-4,
                "oracle_dict": oracle_dict,
                "ngpu": 1,
                "batch_size": 64,
                "seed": 123,
                "weights": "pretrained_models/model/generator2.pth",
                "lambda": 0.01,
                "model": "generator",
                "backbone": biggan_decoder,
                "z_dim": 128,
                "max_epoch": 200,
                "alpha": 0.20,
                "dataset": solid_small}

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
    "return_attributes": False,
}



# Generate dataset
EXP_GROUPS = {}
EXP_GROUPS["dataset"] = hu.cartesian_exp_group({"generator_dict": generator_dict,
                                                "batch_size": 4000,
                                                "oracle_dict": oracle_dict,
                                                "seed": 123,
                                                "dataset": [solid],
                                                "model": "generator",
                                                "dataset_name": "dataset-generated-font-char.h5py"})

# Train classifier
EXP_GROUPS["classifier"] = hu.cartesian_exp_group({"ngpu": 1,
                                                   "weight_decay": 1e-4,
                                                   "seed": 123,
                                                   "lr": 0.01,
                                                   "batch_size": 256,
                                                   "backbone": {"name" :"resnet18"},
                                                   "model": "resnet",
                                                   "max_epoch": 2,
                                                   "dataset": [generated]})
