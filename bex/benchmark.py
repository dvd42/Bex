import os
import pandas as pd
import torch
import copy
import numpy as np
from tqdm import tqdm

from .models import get_model
from .loggers import BasicLogger
from .explainers import get_explainer
from .datasets import get_dataset, DatasetWrapper
from .models.configs import default_configs
from .utils import get_successful_cf, bound_z, normalize_embeddings, compute_metrics


class Benchmark:

    """Main class to evaluate counterfactual explanations methods

    Bex benchmark for evaluating explainability methods. For a list of supported methods see :py:mod:`Bex.explainers`
    The benchmark also supports custom methods, for an example on how to create and evaluate your own methods see :py:class:`ExplainerBase <Bex.explainers.ExplainerBase>`

    Args:
        batch_size (``int``, optional): dataloader batch size (default: 12)
        num_workers (``int``, optional): dataloader number of workers (default: 2)
        n_samples (``int``, optional): number of samples to explain per confidence level (default: 100)
        corr_level (``float``, optional): `0.50` or `0.95` correlation level of the spuriously correlated attributes :math:`z_{\\text{corr}}` (default: 0.95)
        n_corr (``int``, optional): `6` or `10` number of correlated attributes (default: 10)
        seed: (``int``, optional) numpy and torch random seed (default: 0)
        data_path (``str``, optional) path to download the datasets and models, defaults to (~/.bex)
        download(``bool``, optional) download the data for the benchmark if not in available locally. (default: True)
        logger (:py:class:`BasicLogger <Bex.loggers.BasicLogger>`, optional): logger to log results and examples, if `None` nothing will be logged (default: :py:class:`<Bex.loggers.BasicLogger>`)
    """

    def __init__(self, batch_size=12, num_workers=8, n_samples=800, corr_level=0.95, n_corr=10, seed=0, logger=BasicLogger, data_path=None, download=True):

        self.data_path = data_path
        if data_path is None:
            self.data_path = os.path.join(os.path.expanduser("~"), ".bex")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_samples = n_samples
        self.dataset_name = "synbols_font"
        self.corr_level = corr_level
        self.n_corr = n_corr
        self.seed = seed
        self.logger = logger

        self.classifier_name = "resnet"
        self.results = []
        self.current_config = {}

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        dataset = copy.deepcopy(default_configs["dataset"][self.dataset_name])
        dataset["name"] += f"_corr{self.corr_level}_n_clusters{self.n_corr}.h5py"
        print("Loading data...")
        train_set, val_set = get_dataset(["train", "val"], self.data_path, dataset, download=download)
        self.train_dataset = DatasetWrapper(train_set)
        self.val_dataset = DatasetWrapper(val_set)

        generator = get_model("generator", self.data_path, download=download).eval()
        self.encoder = generator.model.embed_attributes
        self.generator = generator

        att = "_" + self.dataset_name.split("_")[1]
        weights = f"_corr{self.corr_level}_n_clusters{self.n_corr}.pth"
        model_name = self.classifier_name + att
        default_configs[model_name]["weights"] = "resnet_font" + weights

        self.classifier = get_model(model_name, self.data_path, download=download).eval()
        self.train_loader = self._get_loader(self.train_dataset, batch_size=512)
        self.val_loader = self._get_loader(self.val_dataset, batch_size=self.batch_size)


    def _set_config(self, explainer, output_path):
        self.current_config["explainer_name"] = explainer if isinstance(explainer, str) else explainer.__name__
        self.current_config["output_path"] = output_path


    def _setup(self, explainer, **kwargs):

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        explainer_name = self.current_config["explainer_name"]
        output_path = self.current_config["output_path"]
        explainer = get_explainer(explainer, self.encoder, self.generator, self.val_loader, **kwargs)
        explainer.data_path = self.data_path
        self._prepare_cache(explainer)
        att = {k: v for k, v in explainer.__dict__.items() if isinstance(v, (str, int, float))}
        if "cache" in att:
            att.pop("cache")

        att.pop("digest")
        run_config = {"explainer": explainer_name,
                      "n_corr": self.n_corr,
                      "corr_level": self.corr_level,
                      "batch_size": self.batch_size,
                      "num_samples": self.n_samples, **att
                     }

        logger = None
        if self.logger is not None:
            logger = self.logger(run_config, output_path)

        return explainer, logger


    def _get_loader(self, dataset, batch_size=12):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers, drop_last=False, shuffle=False)


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


    def _prepare_cache(self, explainer):

        digest = "_".join([self.dataset_name, self.classifier_name, f"corr{self.corr_level}_n_clusters{self.n_corr}", "cache.hdf"])
        cache_dir = os.path.join(self.data_path, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        explainer.digest = os.path.join(cache_dir, digest)

        if not os.path.isfile(explainer.digest):
            print(f"Caching latents to {explainer.digest}")
            explainer._write_cache(self.train_loader, self.encoder, self.generator.model.decode, self.classifier.model, "train")
            explainer._write_cache(self.val_loader, self.encoder, self.generator.model.decode, self.classifier.model, "val")

        explainer._read_cache()


    def _prepare_batch(self, batch, explainer):

        idx, images, labels, categorical_att, continuous_att = batch
        x = images.to(self.device)
        y = labels.to(self.device)
        categorical_att = categorical_att.to(self.device)
        continuous_att = continuous_att.to(self.device)

        latents = explainer._get_latents(idx)
        logits = explainer._get_logits(idx)

        return latents, logits, x, y, categorical_att, continuous_att


    def _get_generator_callable(self, explainer):

        def _generator(latents):

            clipped = latents.clone()
            # clip attributes
            for i in range(latents.shape[-1]):
                lower = explainer.mus_min[i].to(latents.device)
                upper = explainer.mus_max[i].to(latents.device)
                clipped[..., i] = torch.clamp(latents[..., i], min=lower, max=upper)

            clipped = clipped * explainer.latent_std.to(latents.device) + explainer.latent_mean.to(latents.device)
            # binarize inverse color attribute
            clipped[:, -5] = torch.where(clipped[:, -5] < 0.5, 0, 1)


            return self.generator.model.decode(clipped)

        return _generator


    def _cleanup(self, explainer):

        explainer._cleanup()
        self.current_config = {}


    def _accumulate_log(self, logger, metrics, x, decoded, idxs):

        skip = metrics["Cardinality (S#)"] == 0
        to_log = []
        if not skip:
            b, c, h, w = x.size()
            decoded = decoded.view(b, -1, c, h, w)
            to_log = {"samples": torch.tensor([]), "cfs": torch.tensor([])}
            for idx in idxs:
                to_log["samples"] = torch.cat((to_log["samples"], x[idx[0]].detach().cpu()))
                to_log["cfs"] = torch.cat((to_log["cfs"], decoded[idx[0], idx[1]].detach().cpu()))

            to_log["samples"] = to_log["samples"].view(len(idxs), c, h, w)

        logger.accumulate(metrics, to_log)


    def runs(self, exp_list, **kwargs):

        """Evaluates a list of explainers on the Bex benchmark

        Args:
            exp_list (``List[Dict]``): list of dictionaries containing explainers and its parameters
            **kwargs: keyword arguments for :py:meth:`run()`

        Example:
            .. code-block:: python

                bn = bex.Benchmark()
                # run dive and dice
                to_run = [{"explainer": "dive": "lr": 0.1}, {"explainer": "dice": "lr": 0.01}]
                bn.runs(to_run)
        """

        for exp in exp_list:
            self.run(**exp, **kwargs)


    def run(self, explainer, output_path=None, **kwargs):

        """Evaluates an explainer on the Bex benchmark

        Args:
            explainer (``string``): explainability method to be evaluated
            output_path (``string``, optional): directory to store results and examples if `logger` is not `None` (default: output/`datetime.now()`)
            **kwargs: keyword arguments for the explainer :py:mod:`Bex.explainers`

        Example:
            .. code-block:: python

                bn = bex.Benchmark()
                bn.run("stylex")
        """

        self._set_config(explainer, output_path)
        explainer, logger = self._setup(explainer, **kwargs)

        print("Selecting optimal data subset...")
        self._select_data_subset(explainer)
        print(f"Running explainer: {self.current_config['explainer_name']}")

        if logger is None:
            metrics = {"Cardinality": [], "Esc": [], "Ecc": [], "E_causal": [], "E_trivial": []}

        mean = explainer.latent_mean
        std = explainer.latent_std

        embedding_char = self.generator.model.char_embedding.weight.clone()
        embedding_font = self.generator.model.font_embedding.weight.clone()

        char_embed = normalize_embeddings(embedding_char, mean[:3], std[:3])
        font_embed = normalize_embeddings(embedding_font, mean[3:6], std[3:6])

        for batch in tqdm(self.val_loader):

            latents, logits, x, y, categorical_att, continuous_att = self._prepare_batch(batch, explainer)
            b, c = latents.size()

            generator = self._get_generator_callable(explainer)
            z_perturbed = explainer.explain_batch(latents, logits, x, self.classifier.model, generator)
            z_perturbed = z_perturbed.detach()

            z_perturbed = bound_z(z_perturbed, latents, char_embed, font_embed)

            decoded = generator(z_perturbed.view(-1, c))
            logits_perturbed = self.classifier.model(decoded)

            z_perturbed = z_perturbed.view(b, -1, c)
            successful_cf, sce, ncf, cf, trivial = get_successful_cf(z_perturbed, latents, logits_perturbed, logits, char_embed, font_embed)
            cardinality, idxs = compute_metrics(latents, z_perturbed, successful_cf, char_embed, font_embed)

            if logger is not None:
                metrics = {"Cardinality (S#)": cardinality, "SCE":  sce, "Non-Causal Flip": ncf, "Causal FLip": cf, "Trivial": trivial}
                self._accumulate_log(logger, metrics, x, decoded, idxs)
            else:
                metrics["Cardinality (S#)"].append(cardinality)
                metrics["SCE"].append(sce)
                metrics["Non-Causal Flip"].append(ncf)
                metrics["Causal Flip"].append(cf)
                metrics["Trivial"].append(trivial)


        if logger is not None:
            logger.log()
            self.results.append({"explainer": self.current_config["explainer_name"], **logger.metrics})
        else:
            metrics = {k : np.mean(v) for k, v in metrics.items()}
            self.results.append({"explainer": self.current_config["explainer_name"], **metrics})

        self._cleanup(explainer)


    def summarize(self):
        """Summarize the metrics obtained by every explainer ran since :py:class:`Benchmark` was instantiated


        Returns:
            (``pandas.DataFrame``): pandas DataFrame with the results obtained

        """
        return pd.DataFrame(self.results)
