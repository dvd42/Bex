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


encoder_dict = {"lr": 0.001,
                "weight_decay": 1e-4,
                "ngpu": 1,
                "batch_size": 64,
                "seed": 123,
                "model": "encoder",
                "backbone": biggan_encoder,
                "weights": "pretrained_models/model/encoder.pth",
                "z_dim": 128,
                "max_epoch": 200,
                "dataset": solid}

generator_dict = {"lr": 0.001,
                "weight_decay": 1e-4,
                "encoder_dict": encoder_dict,
                "weights": "pretrained_models/model/generator.pth",
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

dive = {"generator_dict": generator_dict,
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
                        "dataset": generated}


default_configs["dataset"] = generated
default_configs["explainer"] = dive
default_configs["classifier"] = classifier_dict
default_configs["encoder"] = encoder_dict
default_configs["generator"] = generator_dict