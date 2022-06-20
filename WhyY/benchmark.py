import os
import wandb
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

    def __init__(self, dataset="synbols_font", data_path="data", batch_size=12, log_images=True, n_samples=100, r=1.,
                 corr_level=0.95, n_clusters_att=10, seed=0):

        self.data_path = data_path
        self.log_images = log_images
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.train_dataset = None
        self.dataset_name = dataset
        self.corr_level = corr_level
        self.n_clusters_att = n_clusters_att
        self.seed = seed
        self._tau = 0.15

        self.r = r
        self.classifier_name = "resnet"
        self.results = []
        self.current_config = {}

        dataset = copy.deepcopy(default_configs["dataset"][self.dataset_name])
        dataset["name"] += f"_corr{self.corr_level}_n_clusters{self.n_clusters_att}.h5py"
        print("Loading data...")
        train_set, val_set = get_dataset(["train", "val"], self.data_path, dataset)
        self.train_dataset = DatasetWrapper(train_set)
        self.val_dataset = DatasetWrapper(val_set)

        # font_2_char = {}
        # font_2_img = {}
        # import matplotlib
        # matplotlib.use("TkAgg")
        # import matplotlib.pyplot as plt
        # for i, x in enumerate(self.train_dataset.dataset.x):
        #     font = self.train_dataset.dataset.raw_labels[i]["font"]
        #     char = self.train_dataset.dataset.raw_labels[i]["char"]
        #     if font not in font_2_char:
        #         font_2_char[font] = []
        #         font_2_img[font] = []
        #     if char not in font_2_char[font]:
        #         font_2_char[font].append(char)

        #         font_2_img[font].append(x)
                # if len(char_2_font[char]) >= 48:
                #     break
            # print(val_dataset.raw_labels[i])
            # print(val_dataset.raw_labels[i]["font"])
            # plt.imshow(x)
            # plt.show()

        # for font in font_2_img:
        #     idxs = np.argsort(font_2_char[font])
        #     font_2_img[font] = np.stack(font_2_img[font])[idxs].tolist()
        # from torchvision.utils import make_grid
        # f, axis = plt.subplots(6, 8)
        # f.set_size_inches(15.5, 15.5)
        # for ax, font in zip(axis.ravel(), sorted(font_2_img.keys())):
        #     images = torch.from_numpy(np.stack(font_2_img[font])).permute(0, 3, 1, 2)
        #     ax.imshow(make_grid(images).permute(1, 2, 0))
        #     ax.axis('off')
        #     ax.set_title(font)
        #     ax.axis('off')
        # plt.show()
        generator = get_model("generator", self.data_path).eval()
        self.encoder = generator.model.embed_attributes
        self.generator = generator


        att = "_" + self.dataset_name.split("_")[1]
        weights = f"_corr{self.corr_level}_n_clusters{self.n_clusters_att}.pth"
        model_name = self.classifier_name + att
        default_configs[model_name]["weights"] = "resnet_font" + weights
        # if not os.path.isfile(os.path.join(self.data_path, default_configs[model_name]["weights"])):
        #     raise FileNotFoundError(f"Weights for {default_configs[model_name]['weights']} are missing")

        # print(f"Loading Classifier from {default_configs[model_name]['weights']}")

        self.classifier = get_model(model_name, self.data_path).eval()
        self.train_loader = self._get_loader(self.train_dataset, batch_size=512)
        self.val_loader = self._get_loader(self.val_dataset, batch_size=self.batch_size)


    def _set_config(self, explainer, log_images, output_path, log_img_thr):
        self.current_config["explainer_name"] = explainer if isinstance(explainer, str) else explainer.__name__
        self.current_config["log_images"] = log_images
        self.current_config["output_path"] = output_path
        self.current_config["log_img_thr"] = log_img_thr


    def _setup(self, explainer, logger, **kwargs):

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        explainer_name = self.current_config["explainer_name"]
        output_path = self.current_config["output_path"]
        log_images = self.current_config["log_images"]
        explainer = get_explainer(explainer, self.encoder, self.generator, self.val_loader, **kwargs)
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
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=False, shuffle=False)


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

        # r = 2 * explainer.mus_max.max() # r = 2
        # r = 10
        # delta = z_perturbed - latents
        # norm = torch.linalg.norm(delta, 1, -1) + 1e-6
        # r = torch.ones_like(norm) * r
        # delta = torch.minimum(norm, r)[:, None] * delta / norm[:, None]

        # continuous attributes
        # r_cont = explainer.mus_max[-6:].max()
        r_cont = self.r
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
            self.type_of_cf["Esc"] = successful_cf.float().mean().item()
            self.type_of_cf["Ecc"].append(((E_cc & ~E_agree).sum() / ncfs).item())
            self.type_of_cf["E_causal"].append(((E_causal_change & ~E_agree).sum() / ncfs).item())
            self.type_of_cf["E_trivial"].append((E_cc & E_agree).float().mean().item())
        else:
            self.type_of_cf["Esc"] = 0
            self.type_of_cf["Ecc"].append(1)
            self.type_of_cf["E_causal"].append(0)

        causal = torch.where(E_causal_change & ~E_agree)
        trivial = torch.where(E_cc & E_agree)
        changes = torch.where(E_cc & ~E_agree)

        return successful_cf, causal, trivial, changes


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
        # char = char / self._tau
        # font = font / self._tau

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

        changes = {"min": 0, "max": 0, "mean": 0, "std": 0}

        if successful_cf.sum() == 0:
            return 0, 0, None, changes

        b, ne, c = z_perturbed.size()
        z = z[:, None, :].repeat(1, ne, 1).view(b, ne, -1).detach()
        test_z = z_perturbed.clone()
        test_l = z.clone()
        z = self._cosine_embedding(z, explainer, binarize=True)
        z_perturbed = self._cosine_embedding(z_perturbed, explainer)

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
                        # index = samples.nonzero().view(-1)[norm_sort][j]
                        # x = self._get_generator_callable(explainer)(test_l[i, index][None])[0]
                        # d = self._get_generator_callable(explainer)(test_z[i, index][None])[0]
                        # print(i, exp[:, 48:96].argmax(1).item(), ortho_set[:, 48:96].argmax(1))
                        # if exp[:, 48:96].argmax(1).item() in ortho_set[:, 48:96].argmax(1):
                        #     import ipdb; ipdb.set_trace()  # BREAKPOINT
                        # import cv2
                        # cv2.imshow("original", x.permute(1, 2, 0).cpu().detach().numpy())
                        # cv2.imshow(f"counterfactual_{j}", d.permute(1, 2, 0).cpu().detach().numpy())
                        # cv2.waitKey(0)


                        idx.append(j)
                        ortho_set = torch.cat((ortho_set, exp), 0)
            # cv2.destroyAllWindows()

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


    def _build_histogram(self, z_perturbed, latents, successful_cf):

        latents = latents[:, None, :].repeat(1, z_perturbed.shape[1], 1)
        z_perturbed = z_perturbed.view_as(latents)
        latents = latents[successful_cf]
        z_perturbed = z_perturbed[successful_cf]
        diff = (z_perturbed - latents).abs()
        self._diff_histogram["char_perturbation"].append(diff[..., :3].mean().item())
        self._diff_histogram["font_perturbation"].append(diff[..., 3:6].mean().item())
        self._diff_histogram["rotation_perturbation"].append(diff[..., -1].mean().item())
        self._diff_histogram["translation-y_perturbation"].append(diff[..., -2].mean().item())
        self._diff_histogram["translation-x_perturbation"].append(diff[..., -3].mean().item())
        self._diff_histogram["scale_perturbation"].append(diff[..., -4].mean().item())
        self._diff_histogram["inverse_color_perturbation"].append(diff[..., -5].mean().item())



    def runs(self, exp_list, **kwargs):
        for exp in exp_list:
            self.run(**exp, **kwargs)


    def run(self, explainer="Dive", logger=WandbLogger, output_path=None, log_images=True, log_img_thr=1, **kwargs):

        self._set_config(explainer, log_images, output_path, log_img_thr)
        explainer, logger = self._setup(explainer, logger, **kwargs)

        print("Selecting optimal data subset...")
        self._select_data_subset(explainer)
        print(f"Running explainer: {self.current_config['explainer_name']}")

        changes = {"min": [], "max": [], "mean": [], "std": []}
        self._diff_histogram = {"font_perturbation": [], "char_perturbation": [], "translation-x_perturbation": [], "translation-y_perturbation": [],
                                "inverse_color_perturbation": [], "scale_perturbation": [], "rotation_perturbation": []}
        self.type_of_cf = {"Ecc": [], "E_causal": [], "E_trivial": []}

        # causal_artifact = wandb.Artifact(f"Causal_5_{self.current_config['explainer_name']}_{self.corr_level}_{self.n_clusters_att}", type="Causal Counterfactuals_5")
        ortho_atrifact = wandb.Artifact(f"Ortho_5_{self.current_config['explainer_name']}_{self.corr_level}_{self.n_clusters_att}", type="Orthogonal Counterfactuals_5")
        # trivial_artifact = wandb.Artifact(f"Trivial_5_{self.current_config['explainer_name']}_{self.corr_level}_{self.n_clusters_att}", type="Trivial Counterfactuals_5")
        # changes_artifact = wandb.Artifact(f"Change_5_{self.current_config['explainer_name']}_{self.corr_level}_{self.n_clusters_att}", type="Change Counterfactuals_5")
        # causal_table = wandb.Table(columns=["ID", "Original", "Counterfactuals", "E_causal"])
        ortho_table = wandb.Table(columns=["ID", "idx", "Original", "Counterfactuals", "E_orthogonal"])
        # trivial_table = wandb.Table(columns=["ID", "Original", "Counterfactuals", "E_Trivial"])
        # changes_table = wandb.Table(columns=["ID", "Original", "Counterfactuals", "E_cc"])
        for i, batch in enumerate(tqdm(self.val_loader)):

            latents, logits, x, y, categorical_att, continuous_att = self._prepare_batch(batch, explainer)
            b, c = latents.size()
            assert torch.allclose(continuous_att.cpu(), latents[:, -5:].cpu() * explainer.latent_std[-5:] + explainer.latent_mean[-5:])

            generator = self._get_generator_callable(explainer)
            with torch.no_grad():
                latents = self.encoder(categorical_att, continuous_att)
                latents = (latents - explainer.latent_mean.cuda()) / explainer.latent_std.cuda()
                logits2 = self.classifier.model(generator(latents))

            assert torch.all(logits2.argmax(1) == logits.argmax(1))
            z_perturbed = explainer.explain_batch(latents, logits, x, self.classifier.model, generator)
            z_perturbed = z_perturbed.detach()


            z_perturbed = self._bound_z(z_perturbed, latents, explainer)

            decoded = generator(z_perturbed.view(-1, c))
            logits_perturbed = self.classifier.model(decoded)

            z_perturbed = z_perturbed.view(b, -1, c)
            successful_cf, causal, trivial, change = self._get_successful_cf(z_perturbed, latents, logits_perturbed, logits, explainer)
            if successful_cf.sum() != 0:
                self._build_histogram(z_perturbed, latents, successful_cf)
            similarity, success, idxs, extra = self._compute_metrics(latents, z_perturbed, successful_cf, explainer)

            from torchvision.utils import make_grid
            import matplotlib.pyplot as plt
            import cv2

            causal_map = {"dive": [2, 5, 31, 36, 32], "stylex": [2, 9, 17, 33, 32, 34, 46, 51, 57],
                          "xgem": [9, 2, 25, 26, 32, 33, 51, 50, 61, 62],
                          "dice": [0, 1, 4, 8, 11, 14, 18, 35, 49, 46, 44, 52, 60, 61]}

            # if causal[0].numel() != 0:
            #     # if i in causal_map[self.current_config["explainer_name"]]:
            #     originals = wandb.Image(make_grid(x[causal[0]]))
            #     causal_cfs = wandb.Image(make_grid(decoded.view(decoded.shape[0] // 10, 10, 3, 32, 32)[causal[0], causal[1]]))
            #     causal_table.add_data(i, originals, causal_cfs, self.type_of_cf["E_causal"])
                    # print(i)
                    # for b, ne in zip(causal[0], causal[1]):
                    #     # for i, (b, ne) in enumerate(zip(trivial[0], trivial[1])):
                    #     o = x[b].permute(1, 2, 0).cpu().detach().numpy()
                    #     d = decoded.view(12, 10, 3, 32, 32)[b, ne].permute(1, 2, 0).cpu().detach().numpy()
                    #     cv2.imshow("original", o)
                    #     cv2.imshow("counterfactual", d)
                    #     cv2.waitKey(0)
                    #     ans = input("save_img?")
                    #     if ans == "y":
                    #         o = o * 0.5 + 0.5
                    #         d = d * 0.5 + 0.5
                    #         o = cv2.cvtColor((o * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
                    #         d = cv2.cvtColor((d * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
                    #         plt.imsave(os.path.join(self.current_config["output_path"], f"original{i}_{b}.png"), o, cmap="gray")
                    #         plt.imsave(os.path.join(self.current_config["output_path"], f"counterfactual{i}_{b}_{ne}.png"), d, cmap="gray")

            if successful_cf.sum() != 0:
                for j, idx in enumerate(idxs):
                # if i in causal_map[self.current_config["explainer_name"]]:
                    originals = wandb.Image(make_grid(x[idx[0]]))
                    orthogonal_cfs = wandb.Image(make_grid(decoded.view(decoded.shape[0] // 10, 10, 3, 32, 32)[idx[0], idx[1]]))
                    ortho_table.add_data(i, j, originals, orthogonal_cfs, success)


            # if trivial[0].numel() != 0:
            #     originals = wandb.Image(make_grid(x[trivial[0]]))
            #     trivial_cfs = wandb.Image(make_grid(decoded.view(decoded.shape[0] // 10, 10, 3, 32, 32)[trivial[0], trivial[1]]))
            #     trivial_table.add_data(i, originals, trivial_cfs, self.type_of_cf["E_trivial"][-1])

            # if change[0].numel() != 0:
            #     originals = wandb.Image(make_grid(x[change[0]]))
            #     changes_cfs = wandb.Image(make_grid(decoded.view(decoded.shape[0] // 10, 10, 3, 32, 32)[change[0], change[1]]))
            #     changes_table.add_data(i, originals, changes_cfs, self.type_of_cf["Ecc"][-1])

                    # cv2.waitKey(0)


            for k, v in extra.items():
                changes[k].append(v)

            metrics = {"similarity": similarity, "success": success}
            # metrics = {"similarity": similarity, "success": success}
            self._accumulate_log(logger, metrics, x, decoded, idxs)

        # causal_artifact.add(causal_table, "Causal counterfactuals")
        ortho_atrifact.add(ortho_table, "Orthogonal counterfactuals")
        # trivial_artifact.add(trivial_table, "Trivial counterfactuals")
        # changes_artifact.add(changes_table, "Change counterfactuals")
        # wandb.run.log_artifact(causal_artifact)
        wandb.run.log_artifact(ortho_atrifact)
        # wandb.run.log_artifact(trivial_artifact)
        # wandb.run.log_artifact(changes_artifact)

        x, y, auc = self._compute_auc(logger.metrics)
        logger.metrics["auc"] = auc
        logger.metrics["auc_x"] = x
        logger.metrics["auc_y"] = y

        logger.metrics.update(changes)
        logger.metrics.update({k: np.mean(v) for k, v in self._diff_histogram.items()})
        logger.metrics.update({k: np.mean(v) for k, v in self.type_of_cf.items()})

        logger.log()
        logger.clean_metrics()
        self.results.append({"explainer": self.current_config["explainer_name"], **logger.metrics, **logger.attributes})
        self._cleanup(explainer, logger)

    def summarize(self):
        return pd.DataFrame(self.results)
