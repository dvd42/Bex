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
                 norm=1, corr_level=0.5, n_clusters_att=2):

        self.data_path = data_path
        self.log_images = log_images
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.train_dataset = None
        self.dataset_name = dataset
        self.corr_level = corr_level
        self.n_clusters_att = n_clusters_att

        self.norm = norm
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
        # import matplotlib
        # matplotlib.use("TkAgg")
        # import matplotlib.pyplot as plt
        # for i, x in enumerate(val_set.dataset.x):
        #     print(val_set.dataset.raw_labels[i]["font"])
        #     print(val_set.dataset.raw_labels[i]["scale"])
        #     print(val_set.dataset.y[i])
        #     plt.imshow(x)
        #     plt.show()

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
        run_config = {"norm": self.norm,
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

        # idx, images, labels, _, _ = batch
        idx, images, labels, categorical_att, continuous_att = batch
        x = images.cuda()
        y = labels.cuda()
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()

        latents = explainer.get_latents(idx)
        logits = explainer.get_logits(idx)

        # import matplotlib.pyplot as plt
        # lambda_range=np.linspace(0,1,10)
        # # fig.subplots_adjust(hspace = .5, wspace=.001)
        # # att_1 = categorical_att[0].unsqueeze(0)
        # # att_1[:, 0] = np.random.choice(list(range(0, 48)))
        # # att_2 = att_1.clone()
        # # att_2[:, 0] = np.random.choice(list(range(0, 48)))
        # att_1 = continuous_att[0].unsqueeze(0)
        # att_2 = att_1.clone()
        # att_1[:, 2] = 0.44
        # att_2[:, 2] = 1.10
        # # idx_to_name = {0: "inverse_color", 1: "pixel_noise_scale", 2:"scale", 3: "translation-x", 4:"translation-y", 5: "rotation"}
        # import matplotlib
        # import cv2
        # imgs = []
        # matplotlib.use("TKAgg")
        # for ind, l in enumerate(lambda_range):
        #     latent_1 = self.generator.model.embed_attributes(categorical_att[0][None], att_1)
        #     latent_2 = self.generator.model.embed_attributes(categorical_att[0][None], att_2)
        #     # latent_1 = self.generator.model.embed_attributes(att_1, continuous_att[0].unsqueeze(0))
        #     # latent_2 = self.generator.model.embed_attributes(att_2, continuous_att[0].unsqueeze(0))
        #     inter_latent = latent_1 * l + (1 - l) * latent_2
        #     inter_image = self.generator.model.decode(inter_latent)
        #     image = inter_image.clamp(0, 1).view(3, 32, 32).permute(1, 2, 0).cpu().detach().numpy()
        #     image = cv2.cvtColor((image * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
        #     imgs.append(image)
        #     # axs[ind].imshow(image, cmap='gray')
        #     # axs[ind].set_title('lambda_val='+str(round(l,1)))
        #     # axs[ind].set_axis_off()
        #     # axs[ind].autoscale(False)
        # # plt.show()
        # imgs = np.stack(imgs)
        # ret = np.zeros((32, 329))
        # for i in range(10):
        #     ret[:, i * 33: i * 33 + 32] = imgs[i]

        # plt.imshow(ret, cmap="gray")
        # plt.show()

        # save = input("Save image?")
        # if save.lower() == "y":
            # ax = plt.gca()
            # ax.axes.yaxis.set_ticklabels([])
            # ax.axes.xaxis.set_ticklabels([])
            # name = input("name?")
            # plt.imsave(name, ret[:, :-2], cmap="gray")


        return latents, logits, x, y, categorical_att, continuous_att


    def _get_generator_callable(self, explainer):

        def _generator(latents):
            latents = latents * explainer.latent_std.cuda() + explainer.latent_mean.cuda()
            color_latent = latents[:, -6]
            color_latent[color_latent > 0.5] = 1
            color_latent[color_latent != 1] = 0
            latents[:, -6] = color_latent

            # total = latents[:, -6].numel()
            # binary = (latents[:, -6] == 0).sum() + (latents[:, -6] == 1).sum()
            # if total != binary:
            #     import pudb; pudb.set_trace()  # BREAKPOINT


            return self.generator.model.decode(latents)

        return _generator

    # bound explanations to lr-ball
    def _bound_z(self, z_perturbed, latents, explainer):

        b, ne , c = z_perturbed.size()
        latents = latents[:, None, :].repeat(1, ne, 1).view(-1, c)
        z_perturbed = z_perturbed.view(-1 ,c)
        r = 2 * explainer.mus_max.max()
        # r = 2
        delta = z_perturbed - latents
        # z_perturbed = latents + torch.clamp(delta, min=-r, max=r)
        norm = torch.linalg.norm(delta, 1, -1) + 1e-6
        r = torch.ones_like(norm) * r
        delta = torch.minimum(norm, r)[:, None] * delta / norm[:, None]

        return z_perturbed.view(b, ne, c)

    # clip continuous attributes
    def _clip_z(self, z_perturbed, explainer):

        for i in range(-6, 0):
            lower = explainer.mus_min[i]
            upper = explainer.mus_max[i]
            z_perturbed[..., i] = torch.clamp(z_perturbed[..., i], min=lower, max=upper)

        return z_perturbed


    def _get_successful_cf(self, z_perturbed, z, perturbed_logits, logits):

        b, ne, c = z_perturbed.size()
        z = z[:, None, :].repeat(1, ne, 1).view(b, ne, c)
        logits = logits[:, None, :].repeat(1, ne, 1).view(b * ne, -1)
        z_perturbed = self._cosine_embedding(z_perturbed)
        z = self._cosine_embedding(z)
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

        # mask1 = perturbed_classifier == classifier
        # mask2 = (perturbed_oracle != oracle) & ~mask1
        # mask3 = (classifier == oracle) & (perturbed_classifier == perturbed_oracle)
        # successful_cf2 = (mask1 | mask2) & ~mask3



        return successful_cf, z_perturbed.view(b, ne, -1)


    def _cosine_embedding(self, z):

        b, ne, c = z.size()
        z = z.view(-1, c)
        weights_char = self.generator.model.char_embedding.weight[None, ...]
        weights_font = self.generator.model.font_embedding.weight[None, ...]
        # first 3 are the embedding of char class
        z_char = z[:, None, :3]
        z_font = z[:, None, 3:259]
        # char = torch.exp((torch.linalg.norm(weights_char - z_char, 2, dim=-1) * -10))
        # font = torch.exp((torch.linalg.norm(weights_font - z_font, 2, dim=-1) * -10))
        # char = torch.softmax((torch.linalg.norm(weights_char - z_char, 2, dim=-1) * -1), dim=-1)
        # font = torch.softmax((torch.linalg.norm(weights_font - z_font, 2, dim=-1) * -1), dim=-1)
        char = torch.cosine_similarity(weights_char, z_char, dim=-1)
        font = torch.cosine_similarity(weights_font, z_font, dim=-1)
        # char_max = torch.cosine_similarity(weights_char, z_char, dim=-1).argmax(-1)
        # font_max = torch.cosine_similarity(weights_font, z_font, dim=-1).argmax(-1)
        # font = torch.zeros(b * ne, 1072).cuda()
        # font[torch.arange(b* ne), font_max] = 1
        z = torch.cat((char, font, z[:, 259:]), 1)

        return z.view(b, ne, -1)

    @torch.no_grad()
    def _compute_metrics(self, z, z_perturbed, successful_cf):

        changes = {"min": 0, "max": 0, "mean": 0, "std": 0}

        if successful_cf.sum() == 0:
            return 0, 0, None, changes

        b, ne, c = z_perturbed.size()
        z = z[:, None, :].repeat(1, ne, 1).view(b, ne, -1).detach()
        z = self._cosine_embedding(z)
        z_perturbed = z_perturbed.view_as(z).detach()


        diff = (z_perturbed - z).abs()
        changes["min"] = diff.min().item()
        changes["max"] = diff.max().item()
        changes["mean"] = diff.mean().item()
        changes["std"] = diff.std().item()

        similarity = []
        success = []
        idxs = []
        correlated = self.val_dataset.dataset.dataset.correlated
        n_corr = len(correlated) // 2
        # correlation = (correlated[:n_corr], correlated[n_corr:])
        # successful_cf[:] = True
        for i, samples in enumerate(successful_cf):

            if z[i][samples].numel() == 0:
                continue

            ortho_set = torch.tensor([]).to(z_perturbed.device)
            norm = torch.linalg.norm(z[i][samples] - z_perturbed[i][samples], ord=self.norm, dim=-1)
            norm_sort = torch.argsort(norm)
            z_perturbed_sorted = z_perturbed[i][norm_sort]
            z_sorted = z[i][norm_sort]

            tau = 0.05
            idx = []
            for j, (exp, latent) in enumerate(zip(z_perturbed_sorted, z_sorted)):
                # exp[-6:] = latent[-6:] - exp[-6:]
                # exp[:48] = latent[:48] - exp[:48]
                # latent = latent - latent.mean()
                exp = latent - exp
                exp = exp[None]
                if ortho_set.numel() == 0:
                    idx.append(j)
                    ortho_set = torch.cat((ortho_set, exp), 0)

                else:

                    cos_sim = torch.cosine_similarity(exp, ortho_set)
                    if torch.all(cos_sim.abs() < tau) or torch.any(cos_sim < -1 + tau):

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

            # import matplotlib.pyplot as plt
            # import matplotlib
            # matplotlib.use("TkAgg")

            # for img, z in zip(x, z_perturbed[:, 0]):
            #     plt.figure()
            #     plt.imshow(img.permute(1, 2, 0).cpu().detach().numpy())
            #     z = generator(z[None])[0]
            #     plt.figure()
            #     plt.imshow(z.permute(1, 2, 0).cpu().detach().numpy())
            #     plt.show()

            self._build_histogram(z_perturbed, latents)
            z_perturbed = self._bound_z(z_perturbed, latents, explainer)
            z_perturbed = self._clip_z(z_perturbed, explainer)

            decoded = generator(z_perturbed.view(-1, c))
            logits_perturbed = self.classifier.model(decoded)

            z_perturbed = z_perturbed.view(b, -1, c)
            successful_cf, z_perturbed = self._get_successful_cf(z_perturbed, latents, logits_perturbed, logits)
            # successful_cf[:] = True
            similarity, success, idxs, extra = self._compute_metrics(latents, z_perturbed, successful_cf)


            for k, v in extra.items():
                changes[k].append(v)

            metrics = {"similarity": similarity, "success": success}
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
            # if successful_cf.sum() != 0:
            #     b, ne, c = z_perturbed.size()
            #     z = latents.repeat(ne, 1).view(-1, c)
            #     # successful_cf[1:5] = True
            #     # successful_cf[-1] = True
            #     z_perturbed = z_perturbed.view(-1, c)
            #     diff = (z_perturbed[successful_cf] - z[successful_cf]).abs()
            #     value, idx = diff.sort(1)
            #     b_coord, z_coord = torch.where(value < 0 + 0.05)

            #     # create a batch with the smallest elements in each tensor incrementally set to 0
            #     # and their corresponding labels
            #     count = torch.unique(b_coord, return_counts=True)[1]
            #     logits = logits.repeat(ne, 1)[successful_cf]
            #     new_z = torch.cat([x.repeat(y, 1) for x, y in zip(z_perturbed[successful_cf], count)])
            #     new_logits = torch.cat([x.repeat(y, 1) for x, y in zip(logits, count)])

            #     start = 0
            #     for c in count:
            #         end = start + c
            #         for j, cf in enumerate(new_z[start: end]):
            #             cf[idx[b_coord, z_coord][start:start + j+1]] = 0.0

                    # start = end
                # ---------------
                # new_successful_cf = self._get_successful_cf(new_z, new_logits)
                # idxs_mask = []
                # start = 0
                # for i, c in enumerate(count):
                #     end = start + c
                #     idxs_mask.append(list(range(start, end)))
                #     start = end

                # new_successful_cf[1] = True
                # new_successful_cf[2] = True
                # new_successful_cf[3] = True
                # new_successful_cf[4] = False
                # new_successful_cf[-1] = True
                # z_perturbed = z_perturbed[successful_cf]
                # idxs = torch.where(new_successful_cf)[0]
                # for i, mask in enumerate(idxs_mask):
                #     for j in idxs:
                #         if j in mask:
                #             z_perturbed[i] = new_z[j]


            # similarity, success = Benchmark._compute_metrics(latents, z_perturbed, successful_cf)



    def summarize(self):
        return pd.DataFrame(self.results)
