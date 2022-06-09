from typing import Iterable
import os
import logging
from datetime import datetime
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import cv2


class BasicLogger:
    """Basic logger class

    :var metrics: dictionary containing the metrics logged by Bex

    This logger logs the metrics computed by the benchmark for a given explainer along with
    the run configuration and some images of successful counterfactuals. It mostly serves as a
    base class for custom loggers but it is fully functional.

    Args:
        attributes (``Dict``): dictionary containing the run config (provided internally)
        path: (``string``): output path for the logger (see :py:meth:`run() <Bex.Benchmark.run>`)
        n_batches: (``int``, optional): max number of image batches to log

    If you wish to test your own explainer on our benchmark use this as a base class and
    override the ``accumulate()`` and ``log()`` methods

    Example:

        .. code-block:: python

            import wandb

            class WandbLogger(BasicLogger):


                def __init__(self, attributes, path, n_batches=10):

                    super().__init__(attributes, path, n_batches)

                    wandb.init(project="Bex", dir=self.path, config=self.attributes, reinit=True)


                def accumulate(self, data, images):

                    super().accumulate(data, images)

                    wandb.log({f"{k}" :v for k, v in data.items()}, commit=True)


                def log(self):

                    self.metrics = {f"{k}_avg": np.mean(v) for k, v in self.metrics.items()}
                    wandb.log(self.metrics)

                    # create a figure with all the images and store it in self._figure
                    self.prepare_images_to_log()

                    wandb.log({"Counterfactuals": self._figure})

            bn = Benchmark()
            bn.run("dive", logger=WandbLogger)

        """

    def __init__(self, attributes, path, n_batches=5):

        self.attributes = attributes
        self.path = path

        self.n_batches = n_batches
        self.metrics = {}
        self.images = []
        self._figure = None

        if self.path is None:
            root = "output"
            now = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            self.path = os.path.join(root, now)
        os.makedirs(self.path, exist_ok=True)


    def accumulate(self, data, images):

        """Updates the metric and image history

        Args:
            data (``Dict``): dictionary containing the metrics compute by Bex
            images (``Dict``): dictionary containing tensor images from a given batch and their orthogonal counterfactuals
        """

        for k, v in data.items():
            if k in self.metrics:
                self.metrics[k].append(v)
            else:
                self.metrics[k] = [v]

        if images:
            if len(self.images) < self.n_batches:
                self.images.append(images)


    def prepare_images_to_log(self):

        """Helper function to build a figure with the accumulated images. Must be called
        before :py:meth:`log() <Bex.loggers.BasicLogger.log>`

        """

        if not self.images:
            logging.addLevelName(logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
            logging.getLogger().setLevel(logging.WARNING)
            logging.warning("There where no successful counterfactuals to log")
            return

        f, ax = plt.subplots(len(self.images), 2, squeeze=False)
        f.set_size_inches(18.5, 10.5)
        for i, batch in enumerate(self.images):
            samples = batch["samples"] * 0.5 + 0.5
            cfs = batch["cfs"] * 0.5 + 0.5

            b = samples.size(0)
            grid = make_grid(samples, nrow=b).permute(1, 2, 0).numpy()
            cfs_grid = make_grid(cfs, nrow=b).permute(1, 2, 0).numpy()
            cfs_grid = cv2.cvtColor(cfs_grid, cv2.COLOR_RGB2GRAY)
            grid = cv2.cvtColor(grid, cv2.COLOR_RGB2GRAY)
            ax[i, 0].imshow(grid, cmap="gray")
            ax[i, 0].set_axis_off()
            ax[i, 0].set_title("Original", fontdict={"fontsize": "small"})
            ax[i, 1].imshow(cfs_grid, cmap="gray")
            ax[i, 1].set_axis_off()
            ax[i, 1].set_title("Counterfactuals", fontdict={"fontsize": "small"})

        self._figure = f


    def log(self):
        """Log the average for each of the metrics computed by Bex along with a figure with
        some of the orthogonal counterfactuals produced.

        The images will the be logged to output_path (see :py:meth:`run() <Bex.Benchmark.run>`) as a
        `.png` file while the metrics along with the run config will be logged as a `.csv` file

        """

        self.metrics = {k: np.mean(v) for k,v in self.metrics.items()}

        self.prepare_images_to_log()
        if self._figure is not None:
            self._figure.savefig(os.path.join(self.path, "counterfactuals.png"), bbox_inches="tight")


        df = pd.DataFrame({**self.metrics, **self.attributes}, index=[0])
        df.to_csv(os.path.join(self.path, "results.csv"))


    def _cleanup(self):

        if self._figure is not None:
            plt.close(self._figure)



