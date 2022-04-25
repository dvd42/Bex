import os
import logging
from datetime import datetime
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import wandb

class BasicLogger:

    def __init__(self, attributes, path, log_images, n_batches=5):

        self.attributes = attributes
        self.n_batches = n_batches
        self.path = path
        self.log_images = log_images
        self.metrics = {}
        self.images = []
        self._figure = None

        if self.path is None:
            root = "output"
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = os.path.join(root, now)
        if not os.path.exists(self.path):
            os.makedirs(self.path)


    def accumulate(self, data, images):

        for k, v in data.items():
            if k in self.metrics:
                self.metrics[k].append(v)
            else:
                self.metrics[k] = [v]

        if self.log_images and images is not None:
            if len(self.images) < self.n_batches:
                self.images.append(images)


    def _prepare_images_to_log(self):

        if not self.images:
            logging.addLevelName(logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
            logging.getLogger().setLevel(logging.WARNING)
            logging.warning("Since no generated counterfactuals exceeded the logging image threshold, no images will be logged for this run. You can modify this threshold via the log_img_thr parameter in the benchmark.run() call")
            return

        f, ax = plt.subplots(len(self.images), 2, squeeze=False)
        f.set_size_inches(18.5, 10.5)
        for i, batch in enumerate(self.images):
            samples = batch["samples"] * 0.5 + 0.5
            cfs = batch["cfs"] * 0.5 + 0.5

        # for i, batch in enumerate(self.tensor_images.chunk(len(self.images))):
            b = samples.size(0)
            grid = make_grid(samples, nrow=1).permute(1, 2, 0).numpy()
            cfs_grid = make_grid(cfs, nrow=b).permute(1, 2, 0).numpy()
            ax[i, 0].imshow(grid)
            ax[i, 0].set_axis_off()
            ax[i, 1].imshow(cfs_grid)
            ax[i, 1].set_axis_off()

        self._figure = f


    def log(self):

        ret = {k: np.mean(v) for k,v in self.metrics.items()}
        self.metrics = ret

        if self.log_images:
            self._prepare_images_to_log()
            self._figure.savefig(os.path.join(self.path, "counterfactuals.png"), bbox_inches="tight")


        pd.DataFrame({**self.attributes, **self.metrics}, index=[0]).to_csv(os.path.join(self.path, "results.csv"))


    def cleanup(self):
        if self._figure is not None:
            plt.close(self._figure)


class WandbLogger(BasicLogger):


    def __init__(self, attributes, path, log_images, n_batches=5):

        super().__init__(attributes, path, log_images, n_batches)

        wandb.init(project="Synbols-benchmark", dir=self.path, config=self.attributes, reinit=True)


    def log(self):

        ret = {k: np.mean(v) for k,v in self.metrics.items()}
        self.metrics = ret
        wandb.log(self.metrics)

        if self.log_images:
            self._prepare_images_to_log()

            wandb.log({"Counterfactuals": self._figure})
