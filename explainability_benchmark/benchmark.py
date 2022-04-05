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

    def __init__(self, dataset="synbols", classifier="resnet", data_path="data", batch_size=12, log_images=True, n_samples=100, load_train=True):

        self.data_path = data_path
        self.log_images = log_images
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.train_dataset = None
        self.dataset_name = dataset
        self.classifier_name = classifier
        self.results = []
        self.current_config = {}
        self.max_norm = 2

        dataset = default_configs["dataset"][self.dataset_name]
        if load_train:
            print("Loading Training data...")
            train_set = DatasetWrapper(get_dataset(["train"], self.data_path, dataset)[0])
            self.train_dataset = train_set
        print("Loading test data...")
        val_set = DatasetWrapper(get_dataset(["val"], self.data_path, dataset)[0])
        self.val_dataset = val_set

        generator = get_model("generator", self.data_path).eval()
        self.encoder = generator.model.embed_attributes
        self.generator = generator

        self.classifier = get_model(self.classifier_name, self.data_path).eval()
        self.val_loader = self._get_loader(self.val_dataset)
        self.train_loader = None
        if load_train:
            self.train_loader = self._get_loader(self.train_dataset)


    def _set_config(self, explainer, log_images, output_path, z_explainer, log_img_thr):
        self.current_config["explainer_name"] = explainer if isinstance(explainer, str) else explainer.__name__
        self.current_config["z_explainer"] = z_explainer
        self.current_config["log_images"] = log_images
        self.current_config["output_path"] = output_path
        self.current_config["log_img_thr"] = log_img_thr


    def _setup(self, explainer, logger, **kwargs):

        z_explainer = self.current_config["z_explainer"]
        explainer_name = self.current_config["explainer_name"]
        output_path = self.current_config["output_path"]
        log_images = self.current_config["log_images"]
        explainer = get_explainer(explainer, self.encoder, self.generator, self.classifier, self.train_loader, z_explainer, **kwargs)
        explainer.data_path = self.data_path
        self._prepare_cache(explainer, z_explainer)
        att = {k: v for k, v in explainer.__dict__.items() if isinstance(v, (str, int, float))}
        if "cache" in att:
            att.pop("cache")

        att.pop("digest")
        run_config = {"classifier": self.classifier_name, "dataset": self.dataset_name, "explainer": explainer_name, "batch_size": self.batch_size, "num_samples": self.n_samples, **att}
        logger = logger(run_config, output_path, log_images=log_images)

        return explainer, logger


    def _get_loader(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=4, drop_last=False, shuffle=False)


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


    def _prepare_cache(self, explainer, z_explainer):

        digest = "_".join([self.dataset_name, self.classifier_name, "cache.hdf"])
        explainer.digest = os.path.join(self.data_path, digest)

        if not os.path.isfile(explainer.digest):
            print("Caching latents")
            if z_explainer:
                # explainer.write_cache(self.train_loader, self.classifier.model, "train")
                explainer.write_cache(self.val_loader, self.classifier.model)

            else:
                explainer.write_cache(self.train_loader, self.encoder, self.generator.model.decode, self.classifier.model, "train")
                explainer.write_cache(self.val_loader, self.encoder, self.generator.model.decode, self.classifier.model, "val")

        if z_explainer:
            explainer.read_cache(self.train_dataset.dataset, self.val_dataset.dataset)
        else:
            explainer.read_cache()


    def _prepare_batch(self, batch, explainer):

        if self.current_config["z_explainer"]:
            idx, x, y = batch
            x = x.cuda()
            y = y.cuda()
        else:
            idx, images, labels, categorical_att, continuous_att = batch
            x = images.cuda()
            y = labels.cuda()
            categorical_att = categorical_att.cuda()
            continuous_att = continuous_att.cuda()

        latents = explainer.get_latents(idx)
        logits = explainer.get_logits(idx)

        return latents, logits, x, y


    def _get_generator_callable(self, explainer):

        def _generator(latents):
            latents = latents * explainer.latent_std - explainer.latent_mean
            return self.generator.model.decode(latents)

        return _generator


    def _boundz(self, z_perturbed):

        total_norm = torch.linalg.norm(z_perturbed, 2, dim=-1)
        clip_coef = torch.clamp(self.max_norm / (total_norm + 1e-6), max=1.0)
        z_perturbed = z_perturbed * clip_coef[..., None]

        return z_perturbed


    def _get_successful_cf(self, z_perturbed, logits):

        b, ne, c = z_perturbed.size()
        z_perturbed = z_perturbed.view(-1, c)
        if self.current_config["z_explainer"]:
            oracle_labels = self.val_dataset.dataset.oracle(z_perturbed)

        else:
            oracle_labels = self.val_dataset.dataset.oracle(z_perturbed, self.generator.model)

        labels = logits.repeat(ne, 1)
        mask = labels.argmax(1).view(b, ne) != oracle_labels.view(b, ne)

        return mask

    @staticmethod
    def _compute_metrics(z, z_perturbed, successful_cf):

        if successful_cf.sum() == 0:
            return 0, 0

        b, num_explanations, c = z_perturbed.size()
        z = z.repeat(num_explanations, 1).view(b, num_explanations, c).detach()
        z_perturbed = z_perturbed.view_as(z).detach()
        z = z[successful_cf]
        z_perturbed = z_perturbed[successful_cf]

        similarity = (F.normalize(z, 1) - F.normalize(z_perturbed, 1)).abs().sum(-1).mean().cpu().numpy()
        similarity = 1 / (1 + similarity)

        # z_perturbed[~successful_cf] = 0
        # success = float((torch.linalg.matrix_rank(z_perturbed) / num_explanations).mean().cpu().numpy())

        ortho_set = torch.tensor([]).to(z_perturbed.device)
        z_perturbed = z_perturbed.view(-1, c)
        norm = torch.linalg.norm(z.view(-1, c) - z_perturbed, 1, dim=1)
        norm_sort = torch.argsort(norm)
        z_perturbed_sorted = z_perturbed[norm_sort]

        eps = 0.05
        for exp in z_perturbed_sorted:
            exp = exp[None]
            if ortho_set.numel() == 0:
                ortho_set = torch.cat((ortho_set, exp), 0)

            else:
                dist = torch.cosine_similarity(exp, ortho_set)
                if torch.all(dist.abs() < 0 + eps):
                    ortho_set = torch.cat((ortho_set, exp), 0)

                elif (dist < -1 + eps).sum() == 1:
                    # non-zero element close to -1
                    ortho_set = torch.cat((ortho_set, exp), 0)

        success = ortho_set.size(0) / z_perturbed_sorted.size(0)

        return similarity, success


    def _cleanup(self, explainer, logger):

        explainer.cleanup()
        logger.cleanup()
        self.current_config = {}


    def _accumulate_log(self, logger, similarity, success, x, decoded):

        to_log = []
        skip = (success + similarity) / 2 < self.current_config["log_img_thr"]
        if self.current_config["log_images"]:
            b, c, h, w = x.size()
            decoded = decoded.view(b, -1, c, h, w)
            to_log = torch.hstack((x[:, None, ...], decoded)).detach().cpu()

        logger.accumulate({"similarity": similarity, "success": success}, to_log, skip=skip)


    def runs(self, exp_list, **kwargs):
        for exp in exp_list:
            self.run(**exp, **kwargs)


    def run(self, explainer="Dive", logger=WandbLogger, z_explainer=False, output_path=None, log_images=True, log_img_thr=0.5, **kwargs):

        self._set_config(explainer, log_images, output_path, z_explainer, log_img_thr)
        explainer, logger = self._setup(explainer, logger, **kwargs)

        print("Selecting optimal data subset...")
        self._select_data_subset(explainer)
        print(f"Running explainer: {self.current_config['explainer_name']}")

        for batch in tqdm(self.val_loader):

            latents, logits, x, y = self._prepare_batch(batch, explainer)

            if self.current_config["z_explainer"]:
                # x and latents are the same thing when working on z-explainer
                z_perturbed = explainer.explain_batch(latents, logits, self.classifier.model)
            else:
                generator = self._get_generator_callable(explainer)
                z_perturbed, decoded = explainer.explain_batch(latents, logits, x, self.classifier.model, generator)

            # l_max_norm bound
            z_perturbed = self._boundz(z_perturbed)

            successful_cf = self._get_successful_cf(z_perturbed, logits)
            similarity, success = Benchmark._compute_metrics(latents, z_perturbed, successful_cf)

            self._accumulate_log(logger, similarity, success, x, decoded)

        logger.log()

        self.results.append({"explainer": self.current_config["explainer_name"], **logger.metrics, **logger.attributes})
        self._cleanup(explainer, logger)


    def summarize(self):
        return pd.DataFrame(self.results)
