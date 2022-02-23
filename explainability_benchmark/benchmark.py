import os
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from .models import get_model
from .explainers import get_explainer
from .datasets import get_dataset
from .models.configs import default_configs
from .tools.loggers import WandbLogger


class DatasetWrapper(torch.utils.data.Dataset):
    """Helper class to provide image id"""

    def __init__(self, dataset, indices=None):
        """Constructor
        Args:
        dataset (torch.utils.data.Dataset): Dataset object
        """
        self.dataset = dataset
        self.indices = indices
        if self.indices is None:
            self.indices = list(range(len(dataset)))

    def __getitem__(self, item):
        return (self.indices[item], *self.dataset[self.indices[item]])

    def __len__(self):
        return len(self.indices)


class Benchmark:

    def __init__(self, data_path="data", batch_size=12, log_images=True, n_samples=100, load_train=True):

        self.data_path = data_path
        self.log_images = log_images
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.train_dataset = None
        self.results = []
        if load_train:
            print("Loading Training data...")
            train_set = DatasetWrapper(get_dataset(["train"], self.data_path, default_configs["dataset"])[0])
            self.train_dataset = train_set
        print("Loading test data...")
        val_set = DatasetWrapper(get_dataset(["val"], self.data_path, default_configs["dataset"])[0])
        self.val_dataset = val_set

        generator = get_model("generator", self.data_path).eval()
        self.encoder = generator.model.embed_attributes
        self.generator = generator

        self.classifier = get_model("resnet", self.data_path).eval()
        self.val_loader = self._get_loader(self.val_dataset)
        self.train_loader = None
        if load_train:
            self.train_loader = self._get_loader(self.train_dataset)



    def _select_data_subset(self, explainer):

        """Instead of using the whole dataset, we use a balanced set of correctly and incorrectly classified samples"""
        if self.n_samples > 0:
            preds = torch.sigmoid(explainer.logits).numpy()
            labels = self.val_dataset.dataset.y.astype(float)
            indices = []
            for confidence in [-0.9, -0.6, -0.4, -0.1, 0.1, 0.4, 0.6, 0.9]:
                # obtain samples that are closest to the required level of confidence
                indices.append(np.abs(labels - preds.max(1) - confidence).argsort()
                                [:self.n_samples])
            indices = np.concatenate(indices, 0)
            self.val_loader.dataset.indices = indices


    def _get_loader(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=4, drop_last=False, shuffle=False)


    @staticmethod
    def _prepare_batch(batch, explainer):

        idx, images, labels, categorical_att, continuous_att = batch
        idx = idx.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()

        latents = explainer.get_latents(idx)
        logits = explainer.get_logits(idx)

        return latents, logits, images, labels


    @staticmethod
    def _compute_metrics(z, z_perturbed, successful_cf):

        b, num_explanations, c = z_perturbed.size()
        z = z.repeat(num_explanations, 1).view(b, num_explanations, c).detach()
        z_perturbed = z_perturbed.view_as(z).detach()

        similarity = 0
        # prevent nans
        if successful_cf.sum() != 0:
            z_norm = F.normalize(z[successful_cf], 1)
            z_perturbed_norm = F.normalize(z_perturbed[successful_cf], 1)
            similarity = (z_norm - z_perturbed_norm).abs().sum(-1).mean().cpu().numpy()
            similarity = 1 / (1 + similarity)

        z_perturbed[~successful_cf] = 0

        success = float((torch.linalg.matrix_rank(z_perturbed) / num_explanations).mean().cpu().numpy())

        return similarity, success

    def _prepare_cache(self, explainer):

        if not os.path.isfile(explainer.digest):
            print("Caching latents")
            explainer.write_cache(self.train_loader, self.encoder, self.generator.model.decode, self.classifier.model, "train")
            explainer.write_cache(self.val_loader, self.encoder, self.generator.model.decode, self.classifier.model, "val")

        explainer.read_cache()


    # TODO get oracle as callable
    def _get_successful_cf(self, decoded, logits):

        b, ne, c, h, w = decoded.size()
        preds = self.generator.oracle(decoded.view(-1, c, h, w))["pred_char"].argmax(1)

        ones = preds % 2 == 1
        oracle_labels = torch.zeros_like(preds)
        oracle_labels[ones] = 1

        labels = logits.repeat(ne, 1)
        mask = labels.argmax(1).view(b, ne) != oracle_labels.view(b, ne)

        return mask


    @staticmethod
    def _cleanup(explainer, logger):

        explainer.cleanup()
        logger.cleanup()

    def summarize(self):

        return pd.DataFrame(self.results)


    def _get_generator_callable(self, explainer):

        def _generator(latents):
            latents = latents * explainer.latent_std - explainer.latent_mean
            return self.generator.model.decode(latents)

        return _generator


    def runs(self, exp_list, **kwargs):
        for exp in exp_list:
            self.run(**exp, **kwargs)


    def run(self, explainer="Dive", logger=WandbLogger, output_path=None, log_images=True, log_img_thr=0.5, **kwargs):

        explainer_name = explainer if isinstance(explainer, str) else type(explainer).__name__
        kwargs["data_path"] = self.data_path
        explainer = get_explainer(explainer, self.encoder, self.generator, self.classifier, self.train_loader, **kwargs)
        self._prepare_cache(explainer)
        att = {k: v for k, v in explainer.__dict__.items() if isinstance(v, (str, int, float))}
        if "cache" in att:
            att.pop("cache")
        run_config = {"explainer": explainer_name, "batch_size": self.batch_size, "num_samples": self.n_samples, **att}
        logger = logger(run_config, output_path, log_images=log_images)

        print("Selecting optimal data subset...")
        self._select_data_subset(explainer)

        print(f"Running explainer: {explainer_name}")
        for batch in tqdm(self.val_loader):

            latents, logits, images, oracle_labels = Benchmark._prepare_batch(batch, explainer)
            b, c, h, w = images.size()

            generator = self._get_generator_callable(explainer)
            z_perturbed, decoded = explainer.explain_batch(latents, logits, images, oracle_labels, self.classifier.model, generator)

            successful_cf = self._get_successful_cf(decoded, logits)
            similarity, success = Benchmark._compute_metrics(latents, z_perturbed, successful_cf)

            to_log = []
            # skip = (successful_cf.sum() / successful_cf.numel()).item() < log_img_thr
            skip = (success + similarity) / 2 < log_img_thr
            if log_images:
                decoded = decoded.view(b, -1, c, h, w)
                to_log = torch.hstack((images[:, None, ...], decoded)).detach().cpu()

            logger.accumulate({"similarity": similarity, "success": success}, to_log, skip=skip)

        logger.log()

        self.results.append({"explainer": explainer_name, **logger.metrics, **logger.attributes})
        Benchmark._cleanup(explainer, logger)
