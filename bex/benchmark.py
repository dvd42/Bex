import os
import pandas as pd
import torch
import copy
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from .models import get_model
from .explainers import get_explainer
from .datasets import get_dataset, DatasetWrapper
from .models.configs import default_configs


class Benchmark:

    """Main class to evaluate counterfactual explanations methods

    Bex benchmark for evaluating explainability methods. For a list of supported methods see :py:mod:`Bex.explainers`
    The benchmark also supports custom methods, for an example on how to create and evaluate your own methods see :py:class:`ExplainerBase <Bex.explainers.ExplainerBase>`

    Args:
        batch_size (``int``, optional): dataloader batch size (default: 12)
        num_workers (``int``, optional): dataloader number of workers (default: 2)
        n_samples (``int``, optional): total number of samples to explain (default: 800)
        corr_level (``float``, optional): `0.50` or `0.95` correlation level of the spuriously correlated attributes :math:`z_{\\text{corr}}` (default: 0.95)
        n_corr (``int``, optional): `6` or `10` number of correlated attributes (default: 10)
        seed: (``int``, optional) numpy and torch random seed (default: 0)
    """

    def __init__(self, batch_size=12, num_workers=2, n_samples=800, corr_level=0.95, n_corr=10, seed=0):

        self.data_path = os.path.join(os.path.expanduser("~"), ".bex")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_samples = n_samples
        self.dataset_name = "synbols_font"
        self.corr_level = corr_level
        self.n_corr = n_corr
        self.seed = seed
        self._tau = 0.15

        self.classifier_name = "resnet"
        self.results = []
        self.current_config = {}

        dataset = copy.deepcopy(default_configs["dataset"][self.dataset_name])
        dataset["name"] += f"_corr{self.corr_level}_n_clusters{self.n_corr}.h5py"
        print("Loading data...")
        train_set, val_set = get_dataset(["train", "val"], self.data_path, dataset)
        self.train_dataset = DatasetWrapper(train_set)
        self.val_dataset = DatasetWrapper(val_set)

        generator = get_model("generator", self.data_path).eval()
        self.encoder = generator.model.embed_attributes
        self.generator = generator


        att = "_" + self.dataset_name.split("_")[1]
        weights = f"_corr{self.corr_level}_n_clusters{self.n_corr}.pth"
        model_name = self.classifier_name + att
        default_configs[model_name]["weights"] = "resnet_font" + weights

        self.classifier = get_model(model_name, self.data_path).eval()
        self.train_loader = self._get_loader(self.train_dataset, batch_size=512)
        self.val_loader = self._get_loader(self.val_dataset, batch_size=self.batch_size)


    def _set_config(self, explainer, output_path):
        self.current_config["explainer_name"] = explainer if isinstance(explainer, str) else explainer.__name__
        self.current_config["output_path"] = output_path


    def _setup(self, explainer, logger, **kwargs):

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

        if logger is not None:
            logger = logger(run_config, output_path)

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
                                [:self.n_samples // 8])
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
        x = images.cuda()
        y = labels.cuda()
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()

        latents = explainer._get_latents(idx)
        logits = explainer._get_logits(idx)

        return latents, logits, x, y, categorical_att, continuous_att


    def _get_generator_callable(self, explainer):

        def _generator(latents):

            clipped = latents.clone()
            # clip attributes
            for i in range(latents.shape[-1]):
                lower = explainer.mus_min[i]
                upper = explainer.mus_max[i]
                clipped[..., i] = torch.clamp(latents[..., i], min=lower, max=upper)

            clipped = clipped * explainer.latent_std.cuda() + explainer.latent_mean.cuda()
            # binarize inverse color attribute
            clipped[:, -5] = torch.where(clipped[:, -5] < 0.5, 0, 1)


            return self.generator.model.decode(clipped)

        return _generator

    # bound perturbations to lr-ball
    def _bound_z(self, z_perturbed, latents, explainer):

        weights_char = self.generator.model.char_embedding.weight.clone()
        weights_font = self.generator.model.font_embedding.weight.clone()
        weights_char = (weights_char - explainer.latent_mean[:3].cuda()) / explainer.latent_std[:3].cuda()
        weights_font = (weights_font - explainer.latent_mean[3:6].cuda()) / explainer.latent_std[3:6].cuda()
        b, ne , c = z_perturbed.size()
        latents = latents[:, None, :].repeat(1, ne, 1).view(-1, c)
        z_perturbed = z_perturbed.view(-1 ,c)
        delta = z_perturbed - latents

        r_cont = 1.
        norm = torch.linalg.norm(delta[:, -5:], 1, -1) + 1e-6
        r = torch.ones_like(norm) * r_cont
        delta[:, -5:] = torch.minimum(norm, r)[:, None] * delta[:, -5:] / norm[:, None]
        delta = torch.minimum(norm, r)[:, None] * delta / norm[:, None]

        # character
        r_char = torch.cdist(weights_char, weights_char, p=1).max()
        norm = torch.linalg.norm(delta[:, :3], 1, -1) + 1e-6
        r = torch.ones_like(norm) * r_char
        delta[:, :3] = torch.minimum(norm, r)[:, None] * delta[:, :3] / norm[:, None]

        # font
        r_font = torch.cdist(weights_font, weights_font, p=1).max()
        norm = torch.linalg.norm(delta[:, 3:6], 1, -1) + 1e-6
        r = torch.ones_like(norm) * r_font
        delta[:, 3:6] = torch.minimum(norm, r)[:, None] * delta[:, 3:6] / norm[:, None]

        z_perturbed = latents + delta

        return z_perturbed.view(b, ne, c)


    def _get_successful_cf(self, z_perturbed, z, perturbed_logits, logits, explainer):

        b, ne, c = z_perturbed.size()
        z = z[:, None, :].repeat(1, ne, 1).view(b, ne, c)
        logits = logits[:, None, :].repeat(1, ne, 1).view(b * ne, -1)
        z_perturbed = self._cosine_embedding(z_perturbed, explainer)
        z = self._cosine_embedding(z, explainer)
        perturbed_preds = z_perturbed.view(-1, z_perturbed.size(-1))[:, :48].argmax(1)
        preds = z.view(-1, z.size(-1))[:, :48].argmax(1)

        oracle = torch.zeros_like(preds)
        perturbed_oracle = torch.zeros_like(preds)
        oracle[preds % 2 == 1] = 1
        perturbed_oracle[perturbed_preds % 2 == 1] = 1

        classifier = logits.argmax(1).view(b, ne)
        perturbed_classifier = perturbed_logits.argmax(1).view(b, ne)
        oracle = oracle.view(b, ne)
        perturbed_oracle = perturbed_oracle.view(b, ne)

        E_cc = perturbed_classifier != classifier
        E_causal_change = ~E_cc & (perturbed_oracle != oracle)
        E_agree = (classifier == oracle) & (perturbed_classifier == perturbed_oracle)
        successful_cf = (E_cc | E_causal_change) & ~E_agree

        ncfs = successful_cf.sum() if successful_cf.sum() !=0 else 1
        if successful_cf.sum() != 0:
            Esc = successful_cf.float().mean().item()
            Ecc = ((E_cc & ~E_agree).sum() / ncfs).item()
            E_causal = ((E_causal_change & ~E_agree).sum() / ncfs).item()
            E_trivial = ((E_cc & E_agree).float().mean().item())
        else:
            Esc = 0
            Ecc = 1
            E_causal = 0
            E_trivial = 0

        return successful_cf, Esc, Ecc, E_causal, E_trivial


    def _cosine_embedding(self, z, explainer, binarize=False):

        b, ne, c = z.size()
        z = z.view(-1, c)
        weights_char = self.generator.model.char_embedding.weight.clone()
        weights_font = self.generator.model.font_embedding.weight.clone()
        weights_char = (weights_char - explainer.latent_mean[:3].cuda()) / explainer.latent_std[:3].cuda()
        weights_font = (weights_font - explainer.latent_mean[3:6].cuda()) / explainer.latent_std[3:6].cuda()
        # first 3 are the embedding of char class
        # the next 3 font embedding
        z_char = z[:, None, :3]
        z_font = z[:, None, 3:6]
        char = torch.softmax((torch.linalg.norm(weights_char[None, ...] - z_char, 2, dim=-1) * -3), dim=-1)
        font = torch.softmax((torch.linalg.norm(weights_font[None, ...] - z_font, 2, dim=-1) * -3), dim=-1)

        if binarize:
            char_max = char.argmax(-1)
            font_max = font.argmax(-1)
            char = torch.zeros(b * ne, 48).cuda()
            char[torch.arange(b * ne), char_max] = 1
            font = torch.zeros(b * ne, 48).cuda()
            font[torch.arange(b* ne), font_max] = 1

        z = torch.cat((char, font, z[:, -5:]), 1)

        return z.view(b, ne, -1)

    @torch.no_grad()
    def _compute_metrics(self, z, z_perturbed, successful_cf, explainer):

        if successful_cf.sum() == 0:
            return 0, None

        b, ne, c = z_perturbed.size()
        z = z[:, None, :].repeat(1, ne, 1).view(b, ne, -1).detach()
        z = self._cosine_embedding(z, explainer, binarize=True)
        z_perturbed = self._cosine_embedding(z_perturbed, explainer)

        z_perturbed = z_perturbed.view_as(z).detach()

        success = []
        idxs = []
        for i, samples in enumerate(successful_cf):

            if z[i][samples].numel() == 0:
                continue

            ortho_set = torch.tensor([]).to(z_perturbed.device)
            norm = torch.linalg.norm(z[i][samples] - z_perturbed[i][samples], ord=1, dim=-1)
            norm_sort = torch.argsort(norm)
            z_perturbed_sorted = z_perturbed[i][norm_sort]
            z_sorted = z[i][norm_sort]

            idx = []
            for j, (exp, latent) in enumerate(zip(z_perturbed_sorted, z_sorted)):

                exp[-5:] = latent[-5:] - exp[-5:]
                exp[:-5] = exp[:-5] * (1 - latent[:-5])
                exp = exp[None]

                if ortho_set.numel() == 0:
                    idx.append(j)
                    ortho_set = torch.cat((ortho_set, exp), 0)

                else:
                    cos_discrete = torch.cosine_similarity(exp[:, :-5], ortho_set[:, :-5])
                    cos_cont = torch.cosine_similarity(exp[:, -5:], ortho_set[:, -5:])
                    cos_sim = cos_discrete + cos_cont
                    if torch.all(cos_sim.abs() < self._tau) or torch.any(cos_sim < -1 + self._tau):

                        idx.append(j)
                        ortho_set = torch.cat((ortho_set, exp), 0)

            success.append(ortho_set.size(0))
            idxs.append((i, samples.nonzero().view(-1)[norm_sort[idx]]))


        return np.mean(success), idxs


    def _cleanup(self, explainer, logger):

        explainer._cleanup()
        if logger is not None:
            logger._cleanup()
        self.current_config = {}


    def _accumulate_log(self, logger, metrics, x, decoded, idxs):

        skip = metrics["E_orthogonal"] == 0
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


    def run(self, explainer, logger=None, output_path=None, **kwargs):
        """Evaluates an explainer on the Bex benchmark

        Args:
            explainer (``string``): explainability method to be evaluated
            logger (:py:class:`BasicLogger <Bex.loggers.BasicLogger>`, optional): logger to log results and examples, if `None` nothing will be logged (default: `None`)
            output_path (``string``, optional): directory to store results and examples if `logger` is not `None` (default: output/`datetime.now()`)
            **kwargs: keyword arguments for the explainer :py:mod:`Bex.explainers`

        Example:
            .. code-block:: python

                bn = bex.Benchmark()
                bn.run("stylex")
        """

        self._set_config(explainer, output_path)
        explainer, logger = self._setup(explainer, logger, **kwargs)

        print("Selecting optimal data subset...")
        self._select_data_subset(explainer)
        print(f"Running explainer: {self.current_config['explainer_name']}")

        if logger is None:
            metrics = {"E_orthogonal": [], "Esc": [], "Ecc": [], "E_causal": [], "E_trivial": []}

        for batch in tqdm(self.val_loader):

            latents, logits, x, y, categorical_att, continuous_att = self._prepare_batch(batch, explainer)
            b, c = latents.size()

            generator = self._get_generator_callable(explainer)
            z_perturbed = explainer.explain_batch(latents, logits, x, self.classifier.model, generator)
            z_perturbed = z_perturbed.detach()

            z_perturbed = self._bound_z(z_perturbed, latents, explainer)

            decoded = generator(z_perturbed.view(-1, c))
            logits_perturbed = self.classifier.model(decoded)

            z_perturbed = z_perturbed.view(b, -1, c)
            successful_cf, Esc, Ecc, E_causal, E_trivial = self._get_successful_cf(z_perturbed, latents, logits_perturbed, logits, explainer)
            success, idxs = self._compute_metrics(latents, z_perturbed, successful_cf, explainer)


            if logger is not None:
                metrics = {"E_orthogonal": success, "Esc": Esc, "Ecc": Ecc, "E_causal": E_causal, "E_trivial": E_trivial}
                self._accumulate_log(logger, metrics, x, decoded, idxs)
            else:
                metrics["E_orthogonal"].append(success)
                metrics["Esc"].append(Esc)
                metrics["Ecc"].append(Ecc)
                metrics["E_causal"].append(E_causal)
                metrics["E_trivial"].append(E_trivial)


        if logger is not None:
            logger.log()
            self.results.append({"explainer": self.current_config["explainer_name"], **logger.metrics})
        else:
            metrics = {k : np.mean(v) for k, v in metrics.items()}
            self.results.append({"explainer": self.current_config["explainer_name"], **metrics})

        self._cleanup(explainer, logger)

    def summarize(self):
        """Summarize the metrics obtained by every explainer ran since :py:class:`Benchmark` was instantiated


        Returns:
            (``pandas.DataFrame``): pandas DataFrame with the results obtained

        """
        return pd.DataFrame(self.results)
