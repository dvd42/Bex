import os
import pandas as pd
import torch
import copy
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import auc
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

    def __init__(self, dataset="synbols_font", data_path="data", batch_size=12, log_images=True, n_samples=100, load_train=True,
                 r=1., corr_level=0.5, n_clusters_att=2):

        self.data_path = data_path
        self.log_images = log_images
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.train_dataset = None
        self.dataset_name = dataset
        self.corr_level = corr_level
        self.n_clusters_att = n_clusters_att
        self._tau = 0.15

        self.r = r
        self.classifier_name = "resnet"
        self.results = []
        self.current_config = {}

        dataset = copy.deepcopy(default_configs["dataset"][self.dataset_name])
        dataset["name"] += f"_corr{self.corr_level}_n_clusters{self.n_clusters_att}.h5py"
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


        att = "_" + self.dataset_name.split("_")[1]
        weights = f"_corr{self.corr_level}_n_clusters{self.n_clusters_att}.pth"
        model_name = self.classifier_name + att
        default_configs[model_name]["weights"] = "models/resnet_font" + weights
        if not os.path.isfile(os.path.join(self.data_path, default_configs[model_name]["weights"])):
            raise FileNotFoundError(f"Weights for {default_configs[model_name]['weights']} are missing")

        self.classifier = get_model(model_name, self.data_path).eval()
        self.val_loader = self._get_loader(self.val_dataset, batch_size=self.batch_size)
        self.train_loader = None
        if load_train:
            self.train_loader = self._get_loader(self.train_dataset, batch_size=512)


    def _set_config(self, explainer, log_images, output_path, log_img_thr):
        self.current_config["explainer_name"] = explainer if isinstance(explainer, str) else explainer.__name__
        self.current_config["log_images"] = log_images
        self.current_config["output_path"] = output_path
        self.current_config["log_img_thr"] = log_img_thr


    def _setup(self, explainer, logger, **kwargs):

        explainer_name = self.current_config["explainer_name"]
        output_path = self.current_config["output_path"]
        log_images = self.current_config["log_images"]
        explainer = get_explainer(explainer, self.encoder, self.generator, self.classifier, self.train_loader, self.val_loader, **kwargs)
        explainer.data_path = self.data_path
        self._prepare_cache(explainer)
        att = {k: v for k, v in explainer.__dict__.items() if isinstance(v, (str, int, float))}
        if "cache" in att:
            att.pop("cache")

        att.pop("digest")
        run_config = {"r": self.r,
                      "n_clusters": self.n_clusters_att,
                      "corr_level": self.corr_level,
                      "dataset": self.dataset_name,
                      "explainer": explainer_name, "batch_size": self.batch_size,
                      "num_samples": self.n_samples, **att
                     }
        logger = logger(run_config, output_path, log_images=log_images)

        return explainer, logger


    def _get_loader(self, dataset, batch_size=12):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=False)


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

        digest = "_".join([self.dataset_name, self.classifier_name, f"corr{self.corr_level}_n_clusters{self.n_clusters_att}", "cache.hdf"])
        cache_dir = os.path.join(self.data_path, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        explainer.digest = os.path.join(cache_dir, digest)

        if not os.path.isfile(explainer.digest):
            print(f"Caching latents to {explainer.digest}")
            explainer.write_cache(self.train_loader, self.encoder, self.generator.model.decode, self.classifier.model, "train")
            explainer.write_cache(self.val_loader, self.encoder, self.generator.model.decode, self.classifier.model, "val")

        explainer.read_cache()


    def _prepare_batch(self, batch, explainer):

        idx, images, labels, categorical_att, continuous_att = batch
        x = images.cuda()
        y = labels.cuda()
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()

        latents = explainer.get_latents(idx)
        logits = explainer.get_logits(idx)

        return latents, logits, x, y, categorical_att, continuous_att


    def _get_generator_callable(self, explainer):

        def _generator(latents):
            latents = latents * explainer.latent_std.cuda() + explainer.latent_mean.cuda()
            clipped = latents.clone()
            for i in range(-6, 0):
                lower = explainer.mus_min[i]
                upper = explainer.mus_max[i]
                clipped[..., i] = torch.clamp(latents[..., i], min=lower, max=upper)
            # color_latent = latents[:, -6]
            # color_latent[color_latent > 0.5] = 1
            # color_latent[color_latent != 1] = 0
            # clipped[:, -6] = color_latent


            return self.generator.model.decode(clipped)

        return _generator

    # bound perturbations to lr-ball
    def _bound_z(self, z_perturbed, latents, explainer):

        weights_char = self.generator.model.char_embedding.weight.clone()
        weights_font = self.generator.model.font_embedding.weight.clone()
        weights_char = (weights_char - explainer.latent_mean[:3].cuda()) / explainer.latent_std[:3].cuda()
        weights_font = (weights_font - explainer.latent_mean[3:259].cuda()) / explainer.latent_std[3:259].cuda()
        b, ne , c = z_perturbed.size()
        latents = latents[:, None, :].repeat(1, ne, 1).view(-1, c)
        z_perturbed = z_perturbed.view(-1 ,c)
        delta = z_perturbed - latents

        # r = 2 * explainer.mus_max.max() # r = 2
        # delta = z_perturbed - latents
        # norm = torch.linalg.norm(delta, 1, -1) + 1e-6
        # r = torch.ones_like(norm) * r
        # delta = torch.minimum(norm, r)[:, None] * delta / norm[:, None]

        # continuous attributes
        # r_cont = explainer.mus_max[-6:].max()
        r_cont = self.r
        norm = torch.linalg.norm(delta[:, -6:], 1, -1) + 1e-6
        r = torch.ones_like(norm) * r_cont
        delta[:, -6:] = torch.minimum(norm, r)[:, None] * delta[:, -6:] / norm[:, None]
        delta = torch.minimum(norm, r)[:, None] * delta / norm[:, None]

        # # # character
        r_char = torch.cdist(weights_char, weights_char, p=1).max()
        norm = torch.linalg.norm(delta[:, :3], 1, -1) + 1e-6
        r = torch.ones_like(norm) * r_char
        delta[:, :3] = torch.minimum(norm, r)[:, None] * delta[:, :3] / norm[:, None]

        # # # font
        r_font = torch.cdist(weights_font, weights_font, p=1).max()
        norm = torch.linalg.norm(delta[:, 3:259], 1, -1) + 1e-6
        r = torch.ones_like(norm) * r_font
        delta[:, 3:259] = torch.minimum(norm, r)[:, None] * delta[:, 3:259] / norm[:, None]

        z_perturbed = latents + delta

        return z_perturbed.view(b, ne, c)

    # clip continuous attributes
    def _clip_z(self, z_perturbed, explainer):

        for i in range(-6, 0):
            lower = explainer.mus_min[i]
            upper = explainer.mus_max[i]
            z_perturbed[..., i] = torch.clamp(z_perturbed[..., i], min=lower, max=upper)

        return z_perturbed


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



        return successful_cf, z_perturbed.view(b, ne, -1)


    def _cosine_embedding(self, z, explainer, binarize=False):

        b, ne, c = z.size()
        z = z.view(-1, c)
        weights_char = self.generator.model.char_embedding.weight.clone()
        weights_font = self.generator.model.font_embedding.weight.clone()
        weights_char = (weights_char - explainer.latent_mean[:3].cuda()) / explainer.latent_std[:3].cuda()
        weights_font = (weights_font - explainer.latent_mean[3:259].cuda()) / explainer.latent_std[3:259].cuda()
        # first 3 are the embedding of char class
        # the next 256 font embedding
        z_char = z[:, None, :3]
        z_font = z[:, None, 3:259]
        char = torch.softmax((torch.linalg.norm(weights_char[None, ...] - z_char, 2, dim=-1) * -1), dim=-1)
        font = torch.softmax((torch.linalg.norm(weights_font[None, ...] - z_font, 2, dim=-1) * -1), dim=-1)
        if binarize:
            char_max = char.argmax(-1)
            font_max = font.argmax(-1)
            char = torch.zeros(b * ne, 48).cuda()
            char[torch.arange(b * ne), char_max] = 1
            font = torch.zeros(b * ne, 1072).cuda()
            font[torch.arange(b* ne), font_max] = 1

        z = torch.cat((char, font, z[:, 259:]), 1)

        return z.view(b, ne, -1)

    @torch.no_grad()
    def _compute_metrics(self, z, z_perturbed, successful_cf, explainer):

        changes = {"min": 0, "max": 0, "mean": 0, "std": 0}

        if successful_cf.sum() == 0:
            return 0, 0, None, changes

        b, ne, c = z_perturbed.size()
        z = z[:, None, :].repeat(1, ne, 1).view(b, ne, -1).detach()
        z = self._cosine_embedding(z, explainer, binarize=True)
        z_perturbed = z_perturbed.view_as(z).detach()


        diff = (z_perturbed - z).abs()
        changes["min"] = diff.min().item()
        changes["max"] = diff.max().item()
        changes["mean"] = diff.mean().item()
        changes["std"] = diff.std().item()

        similarity = []
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

                exp[-6:] = latent[-6:] - exp[-6:]
                exp[:48+1072] = exp[:48+1072] * (1 - latent[:48+1072])
                exp = exp[None]

                if ortho_set.numel() == 0:
                    idx.append(j)
                    ortho_set = torch.cat((ortho_set, exp), 0)

                else:
                    # cos_sim = torch.cosine_similarity(exp, ortho_set)
                    # cos_char = torch.cosine_similarity(exp[:, :48], ortho_set[:, :48])
                    # cos_font = torch.cosine_similarity(exp[:, 48:48+1072], ortho_set[:, 48:48 + 1072])
                    cos_discrete = torch.cosine_similarity(exp[:48+1072], ortho_set[:48+1072])
                    cos_cont = torch.cosine_similarity(exp[:, -6:], ortho_set[:, -6:])
                    cos_sim = cos_discrete + cos_cont
                    if torch.all(cos_sim.abs() < self._tau) or torch.any(cos_sim < -1 + self._tau):

                        idx.append(j)
                        ortho_set = torch.cat((ortho_set, exp), 0)

            success.append(ortho_set.size(0))
            s = norm[norm_sort][idx].mean()
            similarity.append(1 / (1 + s).item())
            idxs.append((i, samples.nonzero().view(-1)[norm_sort[idx]]))


        return np.mean(similarity), np.mean(success), idxs, changes


    def _cleanup(self, explainer, logger):

        explainer.cleanup()
        logger.cleanup()
        self.current_config = {}


    def _accumulate_log(self, logger, metrics, x, decoded, idxs):

        skip = metrics["success"] < self.current_config["log_img_thr"]
        to_log = None
        if self.current_config["log_images"] and not skip:
            b, c, h, w = x.size()
            decoded = decoded.view(b, -1, c, h, w)
            to_log = {"samples": torch.tensor([]), "cfs": torch.tensor([])}
            for idx in idxs:
                to_log["samples"] = torch.cat((to_log["samples"], x[idx[0]].detach().cpu()))
                to_log["cfs"] = torch.cat((to_log["cfs"], decoded[idx[0], idx[1]].detach().cpu()))

            to_log["samples"] = to_log["samples"].view(len(idxs), c, h, w)

        logger.accumulate(metrics, to_log)


    def _compute_auc(self, metrics):

        x = np.linspace(0, 1, 100)
        success = np.array(metrics["success"])
        similarity = np.array(metrics["similarity"])
        y = [success[similarity < thr].mean() for thr in x]

        y = np.nan_to_num(y)

        return x, y, auc(x, y)


    def _build_histogram(self, z_perturbed, latents):

        latents = latents[:, None, :].repeat(1, z_perturbed.shape[1], 1)
        z_perturbed = z_perturbed.view_as(latents)
        diff = (z_perturbed - latents).abs()
        self._diff_histogram["char_perturbation"].append(diff[..., :3].mean().item())
        self._diff_histogram["font_perturbation"].append(diff[..., 3:259].mean().item())
        self._diff_histogram["rotation_perturbation"].append(diff[..., -1].mean().item())
        self._diff_histogram["translation-y_perturbation"].append(diff[..., -2].mean().item())
        self._diff_histogram["translation-x_perturbation"].append(diff[..., -3].mean().item())
        self._diff_histogram["scale_perturbation"].append(diff[..., -4].mean().item())
        self._diff_histogram["pixel_noise_scale_perturbation"].append(diff[..., -5].mean().item())
        self._diff_histogram["inverse_color_perturbation"].append(diff[..., -6].mean().item())



    def runs(self, exp_list, **kwargs):
        for exp in exp_list:
            self.run(**exp, **kwargs)


    def run(self, explainer="Dive", logger=WandbLogger, output_path=None, log_images=True, log_img_thr=1, **kwargs):

        self._set_config(explainer, log_images, output_path, log_img_thr)
        if "method" in kwargs:
            if kwargs["method"] == "none":
                self.current_config["explainer_name"] = "xgem"
        explainer, logger = self._setup(explainer, logger, **kwargs)

        print("Selecting optimal data subset...")
        self._select_data_subset(explainer)
        print(f"Running explainer: {self.current_config['explainer_name']}")

        changes = {"min": [], "max": [], "mean": [], "std": []}
        self._diff_histogram = {"font_perturbation": [], "char_perturbation": [], "translation-x_perturbation": [], "translation-y_perturbation": [],
                                "inverse_color_perturbation": [], "scale_perturbation": [], "rotation_perturbation": [], "pixel_noise_scale_perturbation" : []}

        for batch in tqdm(self.val_loader):

            latents, logits, x, y, categorical_att, continuous_att = self._prepare_batch(batch, explainer)
            b, c = latents.size()

            generator = self._get_generator_callable(explainer)
            with torch.no_grad():
                latents = self.encoder(categorical_att, continuous_att)
                latents = (latents - explainer.latent_mean.cuda()) / explainer.latent_std.cuda()
                logits = self.classifier.model(generator(latents))

            z_perturbed = explainer.explain_batch(latents, logits, x, self.classifier.model, generator)
            z_perturbed = z_perturbed.detach()


            self._build_histogram(z_perturbed, latents)
            z_perturbed = self._bound_z(z_perturbed, latents, explainer)
            # z_perturbed = self._clip_z(z_perturbed, explainer)

            decoded = generator(z_perturbed.view(-1, c))
            logits_perturbed = self.classifier.model(decoded)

            z_perturbed = z_perturbed.view(b, -1, c)
            successful_cf, z_perturbed = self._get_successful_cf(z_perturbed, latents, logits_perturbed, logits, explainer)
            similarity, success, idxs, extra = self._compute_metrics(latents, z_perturbed, successful_cf, explainer)
            n_cfs = (successful_cf.sum() / successful_cf.numel()).item()


            for k, v in extra.items():
                changes[k].append(v)

            metrics = {"similarity": similarity, "success": success, "n_cfs": n_cfs}
            # metrics = {"similarity": similarity, "success": success}
            self._accumulate_log(logger, metrics, x, decoded, idxs)

        x, y, auc = self._compute_auc(logger.metrics)
        logger.metrics["auc"] = auc
        logger.metrics["auc_x"] = x
        logger.metrics["auc_y"] = y

        logger.metrics.update(changes)
        logger.metrics.update({k: np.mean(v) for k, v in self._diff_histogram.items()})


        logger.log()
        logger.clean_metrics()
        self.results.append({"explainer": self.current_config["explainer_name"], **logger.metrics, **logger.attributes})
        self._cleanup(explainer, logger)

    def summarize(self):
        return pd.DataFrame(self.results)
