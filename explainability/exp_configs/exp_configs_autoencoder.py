generated = {
    "backend": "generated_synbols",
    "n_continuous": 6,
    "width": 32,
    "height": 32,
    "channels": 3,
    "name": "dataset-generated_large.h5py",
    "num_classes": 2,
    "augmentation": False,
    "mask": "random",
    "return_attributes": False,
}



# Generate dataset
EXP_GROUPS["dataset"] = hu.cartesian_exp_group({"model_weights": "model_final.pth",
                                                "ngpu": 1,
                                                "weight_decay": 1e-4,
                                                "seed": 123,
                                                "lambda": 0.001,
                                                "alpha": 0.01,
                                                "lr": 0.01,
                                                "batch_size": 5000,
                                                "model": "autoencoder",
                                                "max_epoch": 200,
                                                "backbone": [biggan_biggan],
                                                "z_dim": [128],
                                                "dataset": [solid],
                                                "dataset_path": "dataset-generated_large.h5py"})


# Train classifier
EXP_GROUPS["classifier"] = hu.cartesian_exp_group({"ngpu": 1,
                                                   "weight_decay": 1e-4,
                                                   "seed": 123,
                                                   "lr": 0.01,
                                                   "batch_size": 256,
                                                   "backbone": {"name" :"resnet18"},
                                                   "model": "resnet",
                                                   "max_epoch": 1,
                                                   "dataset": [generated]})
