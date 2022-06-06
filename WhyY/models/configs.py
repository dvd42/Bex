import copy
default_configs = {}


solid = {
    "backend": "synbols_hdf5",
    "n_continuous": 5,
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "datasets/non-camou-bw_n=100000_2022-May-31.h5py",
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


encoder_dict = {"lr": 0.001,
                "weight_decay": 1e-4,
                "ngpu": 1,
                "batch_size": 64,
                "seed": 123,
                "model": "encoder",
                "backbone": biggan_encoder,
                "weights": "encoder.pth",
                "z_dim": 3,
                "max_epoch": 100,
                "dataset": solid}

generator_dict = {"lr": 0.001,
                "weight_decay": 1e-4,
                "encoder_dict": encoder_dict,
                "weights": "generator.pth",
                "ngpu": 1,
                "batch_size": 64,
                "seed": 123,
                "lambda": 0.01,
                "model": "generator",
                "backbone": biggan_decoder,
                "z_dim": 3,
                "max_epoch": 100,
                "alpha": 0.20,
                "dataset": solid}

generated_font_corr = {
    "backend": "generated_synbols",
    "n_continuous": 5,
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "dataset_font",
    "num_classes": 2,
    "augmentation": False,
    "mask": "random",
    "n_attributes": 262,
    "return_attributes": True}



resnet_dict_font = {"ngpu": 1,
                   "weight_decay": 1e-4,
                   "seed": 123,
                   "lr": 0.01,
                   "batch_size": 256,
                   "backbone": {"name" :"resnet18"},
                   # "weights":None,
                   "weights":"resnet_font",
                   "model": "resnet",
                   "max_epoch": 10,
                    "dataset": generated_font_corr}



default_configs["dataset"] = {"synbols_font": generated_font_corr}
                              # "synbols_scale": generated_scale_corr}
default_configs["resnet_font"] = resnet_dict_font
default_configs["encoder"] = encoder_dict
default_configs["generator"] = generator_dict
