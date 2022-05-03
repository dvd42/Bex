import copy
default_configs = {}


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

generated_font_corr = {
    "backend": "generated_synbols",
    "n_continuous": 6,
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "dataset-font-correlation.h5py",
    "num_classes": 2,
    "augmentation": False,
    "mask": "random",
    "return_attributes": True}


generated_color_corr = copy.deepcopy(generated_font_corr)
generated_color_corr["name"] = "dataset-color-correlation.h5py"

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
                "z_dim": 128,
                "max_epoch": 200,
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
                "z_dim": 128,
                "max_epoch": 200,
                "alpha": 0.20,
                "dataset": solid}


resnet_dict_font = {"ngpu": 1,
                   "weight_decay": 1e-4,
                   "seed": 123,
                   "lr": 0.01,
                   "batch_size": 256,
                   "backbone": {"name" :"resnet18"},
                   "weights":None,
                   "model": "resnet",
                   "max_epoch": 2,
                    "dataset": generated_font_corr}


resnet_dict_color = copy.deepcopy(resnet_dict_font)
resnet_dict_color["dataset"] = generated_color_corr
resnet_dict_color["weights"] = None


mlp_dict = {"ngpu": 1,
            "weight_decay": 1e-4,
            "seed": 123,
            "lr": 0.01,
            "batch_size": 256,
            "backbone": {"name" :"mlp", "n_hidden": 128, "num_layers": 2},
            "weights": "classifier_toy.pth",
            "max_epoch": 2,
            }


default_configs["dataset"] = {"synbols_font": generated_font_corr,
                              "synbols_color": generated_color_corr}
default_configs["resnet_font"] = resnet_dict_font
default_configs["resnet_color"] = resnet_dict_color
default_configs["mlp"] = mlp_dict
default_configs["encoder"] = encoder_dict
default_configs["generator"] = generator_dict
