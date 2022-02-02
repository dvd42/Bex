from abc import abstractmethod
import os
import json
import sys
import time
from copy import deepcopy
from json import decoder
from os.path import join

import cv2
import h5py
import numpy as np
import pylab
import pylab as pl
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from haven import haven_utils as hu
from numpy.lib.function_base import vectorize
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
import sys
sys.path.append("..")
from models import get_model
from datasets import get_dataset

# from src import datasets, models, wrappers
# from src.models.biggan import Decoder
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

idx_to_name = {0: "inverse_color", 1: "pixel_noise_scale", 2:"scale", 3: "translation-x", 4:"translation-y", 5: "rotation"}

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


class ExplainerBase(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def attack_batch(self, oracle, generator):
        pass


class GradientAttack(ExplainerBase):
    """Main class to generate counterfactuals"""

    def __init__(self, exp_dict, savedir, data_path):
        """Constructor
        Args:
        exp_dict (dict): hyperparameter dictionary
        savedir (str): root path to experiment directory
        data_path (str): root path to datasets and pretrained models
        """
        super().__init__()
        self.exp_dict = exp_dict
        self.savedir = savedir
        self.generator_dict = self.exp_dict["generator_dict"]
        self.generator_path = self.generator_dict["weights"]
        self.classifier_dict = self.exp_dict["classifier_dict"]
        self.classifier_path = self.classifier_dict["weights"]
        train_set = DatasetWrapper(get_dataset(["train"], data_path, self.exp_dict)[0])
        val_set = DatasetWrapper(get_dataset(["val"], data_path, self.exp_dict)[0])
        self.train_dataset = train_set
        self.val_dataset = val_set
        # self.current_attribute = self.dataset.dataset.all_attributes.index(self.exp_dict['attribute'])
        self.data_path = data_path
        self.classifier = self.load_classifier()
        self.generator = self.load_generator()
        self.oracle = self.generator.oracle
        self.appended_images = 0

        if self.exp_dict.get("cache_only", False):
            self.read_or_write_cache()
        else:
            self.read_or_write_cache()
            self.select_data_subset()
            # self.attack_dataset()

    def load_classifier(self):
        """Helper function to load the classifier model"""

        print("Loading classifier...")
        classifier = get_model(self.classifier_dict)
        classifier.load_state_dict(hu.torch_load(self.classifier_path))
        classifier.eval().cuda()
        return classifier


    def load_generator(self):
        """Helper function to load the generator model"""

        print("Loading generator...")
        generator = get_model(self.generator_dict)
        generator.load_state_dict(hu.torch_load(self.generator_path))
        generator.eval().cuda()
        return generator


    def get_loader(self, batch_size, mode):
        """Helper function to create a dataloader

            Args:
            batch_size (int): the batch_size

            Returns:
            torch.utils.data.DataLoader: dataloader without shuffling and chosen bs
        """
        dataset = getattr(self, f"{mode}_dataset")
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=False, shuffle=False)


    def read_or_write_cache(self):

        """Compute and store the Fisher Information Matrix
            for future use as well as other auxiliary data

            Raises:

            TimeoutError: if two processes collide, one will wait for the other to finish to avoid repeating operations.
            If the wait is too long, it raises an error.
        """
        self.digest = f"cache_{hu.hash_dict(self.classifier_dict)}_{hu.hash_dict(self.generator_dict)}"
        self.digest = os.path.join(self.data_path, self.digest)

        if self.exp_dict.get("force_cache", False):
            os.remove(self.digest)

        if not os.path.isfile(self.digest):
        # if not os.path.isfile(self.digest):
        #     try:
        #         os.makedirs(f"{self.digest}.lock")
        #         lock = False
        #     except FileExistsError:
        #         lock = True
        #     if lock:
        #         print(f"Waiting for another process to finish computing cache or delete {self.digest}.lock to continue")
        #         t = time.time()
        #         while os.path.isdir(f"{self.digest}.lock"):
        #             if (time.time() - t) / 3600 > 2:
        #                 raise TimeoutError(f"Timout while waiting for another process to \
        #                                     finish computing cache on {self.digest}. Delete \
        #                                     if that is not the case")
        #             time.sleep(1)
        #         lock = False
            # self.write_cache()
            print("Caching FIM")
            self.write_cache("train")
            print("Caching latents")
            self.write_cache("val")

        self.read_cache()


    def select_data_subset(self):
        """Instead of using the whole dataset, we use a balanced set of correctly and incorrectly classified samples"""
        if self.exp_dict["n_samples"] > 0:
            preds = torch.sigmoid(self.logits).numpy()
            labels = self.val_dataset.dataset.y.astype(float)
            indices = []
            for confidence in [-0.9, -0.6, -0.4, -0.1, 0.1, 0.4, 0.6, 0.9]:
                # obtain samples that are closest to the required level of confidence
                indices.append(np.abs(labels - preds.max(1) - confidence).argsort()
                                [:self.exp_dict["n_samples"]])
            indices = np.concatenate(indices, 0)
            self.val_dataset.indices = indices


    def write_cache(self, mode):
        """Loops through the data and stores latents and FIM"""

        loader = self.get_loader(self.exp_dict['cache_batch_size'], mode)
        preds = []
        fishers = 0
        mus = []
        for idx, x, y, categorical_att, continuous_att in tqdm(loader):
            with torch.no_grad():
                x = x.cuda()
                labels = y.cuda()
                categorical_att = categorical_att.cuda()
                continuous_att = continuous_att.cuda()
                z = self.generator.model.embed_attributes(categorical_att, continuous_att)
                mus.append(z.cpu().numpy())
                # outputs = self.model.model.encode(x)
                # mu = outputs["z"]
                # mus.append(mu)
                # logvars.append(logvar.cpu())
                # logvars.append(0)
                # b, c = mu.size()
                # z = mu.data.clone()
                self.first = True

            def jacobian_forward(z):

                reconstruction = self.generator.model.decode(z)
                logits = self.classifier.model(reconstruction)
                if self.first:
                    preds.append(logits.data.cpu().numpy())
                    self.first = False
                y = torch.distributions.Bernoulli(logits=logits).sample().detach()
                logits = logits * y + (1 - logits) * (1 - y)
                loss = logits.sum(0)

                return loss

            grads = torch.autograd.functional.jacobian(jacobian_forward, z)
            if mode == "train":
                b, c = z.size()
                num_classes = self.exp_dict["dataset"]["num_classes"]

                with torch.no_grad():
                    fisher = torch.matmul(grads[:, :, :, None], grads[:, :, None, :]).view(num_classes, b, c, c).sum(1).cpu()
                fishers += fisher.numpy()
                del(fisher)
                to_save = dict(fishers=fishers)
            elif mode == "val":
                to_save = dict(mus=np.concatenate(mus, 0),
                               logits=np.concatenate(preds, 0))
            del(z)

        with h5py.File(self.digest, 'a') as outfile:
            for k, v in to_save.items():
                outfile[k] = v
                # os.removedirs(f"{self.digest}.lock")
                print("Done.")


    def read_cache(self):
        """Reads cached data from disk"""
        print("Loading from %s" % self.digest)
        self.loaded_data = h5py.File(self.digest, 'r')
        try:
            self.fisher = torch.from_numpy(self.loaded_data['fishers'][...])
            self.logits = torch.from_numpy(self.loaded_data['logits'][...])
            self.mus = torch.from_numpy(self.loaded_data['mus'][...])
        except:
            print("FIM not found in the hdf5 cache")
            print("Done.")


    def attack_batch(self, batch):
        """Uses gradient descent to compute counterfactual explanations
        Args:
        batch (tuple): a batch of image ids, images, and labels
        Returns:
            dict: a dictionary containing the whole attack history
        """
        idx, images, labels, categorical_att, continuous_att = batch
        idx = idx.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()

        predicted_labels = torch.sigmoid(self.logits[idx])

        predicted_labels = predicted_labels.float().cuda()

        diversity_weight = self.exp_dict['diversity_weight']
        lasso_weight = self.exp_dict['lasso_weight']
        reconstruction_weight = self.exp_dict['reconstruction_weight']

        import matplotlib.pyplot as plt
        lambda_range=np.linspace(0,1,10)
        fig, axs = plt.subplots(2,5, figsize=(15, 6))
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()
        att_1 = categorical_att[0].unsqueeze(0)
        att_1[:, 0] = np.random.choice(list(range(0, 48)))
        att_2 = att_1.clone()
        att_2[:, 0] = np.random.choice(list(range(0, 48)))
        cont_1 = continuous_att[0].unsqueeze(0)
        cont_2 = cont_1.clone()
        cont_1[0][5] = 0
        cont_2[0][5] = 1
        import matplotlib
        matplotlib.use("TKAgg")
        for ind, l in enumerate(lambda_range):
            latent_1 = self.generator.model.embed_attributes(att_1, continuous_att[0].unsqueeze(0))
            latent_2 = self.generator.model.embed_attributes(att_2, continuous_att[0].unsqueeze(0))
            inter_latent = latent_1 * l + (1 - l) * latent_2
            inter_image = self.generator.model.decode(inter_latent)
            image = inter_image.clamp(0, 1).view(3, 32, 32).permute(1, 2, 0).cpu().detach().numpy()
            axs[ind].imshow(image, cmap='gray')
            axs[ind].set_title('lambda_val='+str(round(l,1)))
        plt.show()
        fig, axs = plt.subplots(2,5, figsize=(15, 6))
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()
        for ind, l in enumerate(lambda_range):
            latent_1 = self.generator.model.embed_attributes(categorical_att[0].unsqueeze(0), cont_1)
            latent_2 = self.generator.model.embed_attributes(categorical_att[0].unsqueeze(0), cont_2)
            inter_latent = latent_1 * l + (1 - l) * latent_2
            inter_image = self.generator.model.decode(inter_latent)
            image = inter_image.clamp(0, 1).view(3, 32, 32).permute(1, 2, 0).cpu().detach().numpy()
            axs[ind].imshow(image, cmap='gray')
            axs[ind].set_title('lambda_val='+str(round(l,1)))
        plt.show()
        latents = self.mus[idx].cuda()
        b, c = latents.size()

        num_explanations = self.exp_dict['num_explanations']
        epsilon = torch.randn(b, num_explanations, c, requires_grad=True, device=latents.device)
        epsilon.data *= 0.01

        mask = self.get_mask(batch, latents)

        optimizer = torch.optim.Adam([epsilon], lr=self.exp_dict['lr'], weight_decay=0)

        attack_history = []

        class DecoderClassifier(torch.nn.Module):
            def __init__(self, g, c):
                super().__init__()
                self.g = g
                self.c = c

            def forward(self, x):
                decoded = self.g(x)
                return decoded, self.c(decoded)

        decoder_classifier = DecoderClassifier(self.generator.model.decoder, self.classifier.model)

        for it in range(self.exp_dict['max_iters']):
            optimizer.zero_grad()
            div_regularizer = 0
            lasso_regularizer = 0
            reconstruction_regularizer = 0

            repeat_dim = epsilon.size(0) // mask.size(0)
            epsilon.data = epsilon.data * mask.repeat(repeat_dim, 1, 1)
            # epsilon.data = epsilon.data * mask
            z_perturbed = latents[:, None, :].detach() + epsilon
            if diversity_weight > 0:
                epsilon_normed = epsilon
                epsilon_normed = F.normalize(epsilon_normed, 2, -1)
                div_regularizer = torch.matmul(epsilon_normed, epsilon_normed.permute(0, 2, 1))
                div_regularizer = div_regularizer * (1 - torch.eye(div_regularizer.shape[-1],
                                                                    dtype=div_regularizer.dtype,
                                                                    device=div_regularizer.device))[None, ...]
                div_regularizer = (div_regularizer ** 2).sum()

            decoded, logits = decoder_classifier(z_perturbed.view(b * num_explanations, c))
            bn, ch, h, w = decoded.size()
            if reconstruction_weight > 0:
                reconstruction_regularizer = torch.abs(images[:, None, ...] - decoded.view(b, num_explanations, ch, h, w)).sum()

            lasso_regularizer = torch.abs(z_perturbed - latents[:, None, :]).sum()

            regularizer = lasso_regularizer * lasso_weight + \
                                    div_regularizer * diversity_weight + \
                                    reconstruction_regularizer * reconstruction_weight

            # regularizer = regularizer / mask.expand_as(z_perturbed).sum()
            regularizer = regularizer / mask.repeat(repeat_dim, 1, 1).sum()

            # loss_attack = F.binary_cross_entropy_with_logits(logits, 1 - predicted_labels.repeat(num_explanations, 1), reduction='none')
            loss_attack = F.binary_cross_entropy_with_logits(logits, 1 - predicted_labels.repeat(num_explanations, 1))
            # loss_attack = (loss_attack.view(b, num_explanations) * opt_mask).mean()
            loss = loss_attack + regularizer
            loss.backward()
            optimizer.step()
            attack_history = dict(
                                iter=np.array([it]),
                                idx=idx.data.cpu().numpy(),
                                logits=logits.data.cpu().view(b, num_explanations, -1).numpy(),
                                labels=np.array(labels.squeeze().data.cpu().numpy()),
                                loss=np.array([float(loss)]),
                                loss_attack=np.array([float(loss_attack)]),
                                reconstruction_regularizer=np.array([float(reconstruction_regularizer)]),
                                div_regularizer=np.array([float(div_regularizer)]),
                                lasso_regularizer=np.array([float(lasso_regularizer)]),
                                )  # instead of append, directly set dict due to memory constraints
            success_rate = float((logits.argmax(1) != predicted_labels.repeat(num_explanations, 1).argmax(1)).float().mean())
            attack_history['success_rate'] = np.array([success_rate])
            if success_rate >= self.exp_dict["stop_batch_threshold"]:
                break

        pred_dict = {}
        with torch.no_grad():
            torch.cuda.empty_cache()
            y_perturbed = self.classifier.model(decoded)
            oracle_preds_attack = self.oracle(decoded)

            pred_dict["font"] = oracle_preds_attack["pred_font"].argmax(1).view(b, num_explanations).cpu().tolist()
            pred_dict["char"] = oracle_preds_attack["pred_char"].argmax(1).view(b, num_explanations).cpu().tolist()
            # set continous attributes dict
            for k, v in idx_to_name.items():
                pred_dict[v] = oracle_preds_attack["pred_continuous"][:, k].view(b, num_explanations).cpu().tolist()

            attack_history["oracle_preds_attack"] = json.dumps(pred_dict)

            pred_dict["font"] = categorical_att[:, -1].cpu().tolist()
            pred_dict["char"] = categorical_att[:, -2].cpu().tolist()

            for k, v in idx_to_name.items():
                pred_dict[v] = continuous_att[:, k].cpu().tolist()

            attack_history["oracle_preds_reconstruction"] = json.dumps(pred_dict)

            attack_history["classifier_preds_attack"] = y_perturbed.cpu().numpy()
            attack_history["attacked_latents"] = z_perturbed.cpu().numpy()
            attack_history['latent_similarity'] = (F.normalize(latents, 2, -1).repeat(num_explanations, 1).view(b, num_explanations, -1) *
                                                    F.normalize(z_perturbed, 2, -1).view(b, num_explanations, -1)).sum(-1).view(b, -1).cpu().numpy()
            if self.appended_images < 5:
                attack_history["decoded"] = decoded.view(b, num_explanations, *images.size()[1:]).cpu().numpy()
                attack_history["images"] = images.cpu().numpy()
                self.appended_images += 1

        return attack_history


    def attack_dataset(self):
        """Loops over the dataset and generates counterfactuals for all the samples"""

        loader = self.get_loader(self.exp_dict['batch_size'] // self.exp_dict['num_explanations'], mode="val")
        attack_histories = {}
        for batch in tqdm(loader):
            history = self.attack_batch(batch)
            for k, v in history.items():
                if k in attack_histories:
                    attack_histories[k].append(v)
                else:
                    attack_histories[k] = [v]
            # if self.appended_images == 5:
            #     break
            # print(f"total_loss: {history['loss']},",
            #                     f"loss_attack: {history['loss_attack']},",
            #                     f"reconstruction: {history['reconstruction_regularizer']},",
            #                     f"lasso: {history['lasso_regularizer']},",
            #                     f"diversity: {history['div_regularizer']},",
            #                     f"success_rate: {history['success_rate']}")

        return attack_histories


    def get_mask(self, batch, latents):
        """Helper function that outputs a binary mask for the latent
            space during the counterfactual explanation
        Args:
            latents (torch.Tensor): dataset latents (precomputed)
        Returns:
            torch.Tensor: latents mask
        """
        method = self.exp_dict["method"]
        num_explanations = self.exp_dict['num_explanations']

        if 'fisher' in method:
            if self.exp_dict['fisher_samples'] <= 0:
                # fishers = [self.fisher[self.current_attribute]]
                fishers = self.fisher
            else:
                fishers = self.get_pointwise_fisher(batch, self.exp_dict['fisher_samples'])

        if method in ["fisher_chunk"]:
            masks = []
            for fisher in fishers:
                indices = torch.diagonal(fisher).argsort(descending=True, dim=-1)
                mask = torch.ones(num_explanations, latents.shape[1], device=latents.device, dtype=latents.dtype)
                chunk_size = latents.shape[1] // num_explanations
                for i in range(num_explanations):
                    mask.data[i, indices[(i * chunk_size) :((i + 1) * chunk_size)]] = 0
                masks.append(mask)
            mask = torch.stack(masks, 0)
        elif method in ["fisher_range"]:
            masks = []
            for fisher in fishers:
                indices = torch.diagonal(fisher).argsort(descending=True, dim=-1)
                mask = torch.ones(num_explanations, latents.shape[1], device=latents.device, dtype=latents.dtype)
                for i in range(num_explanations):
                    mask.data[i, indices[0:i]] = 0
                masks.append(mask)
            mask = torch.stack(masks, 0)
        elif method in ["fisher_spectral", "fisher_spectral_inv"]:
            masks = []
            for fisher in fishers:
                scluster = SpectralClustering(n_clusters=num_explanations, affinity='precomputed',
                                                assign_labels='discretize', random_state=0, eigen_solver='arpack', eigen_tol=1e-6)
                affinity = fisher.numpy()
                affinity = affinity - affinity.min()
                affinity /= affinity.max()
                scluster.fit(affinity)
                mask = torch.zeros(num_explanations, latents.shape[1], device=latents.device)
                for i in range(num_explanations):
                    mask[i, torch.from_numpy(scluster.labels_).to(latents.device) == i] = 1
                masks.append(mask)
            mask = torch.stack(masks, 0)
            if 'inv' in method:
                mask = 1 - mask
        elif method in ["random"]:
            indices = torch.randperm(latents.shape[1], device=latents.device)
            mask = torch.ones(num_explanations, latents.shape[1], device=latents.device, dtype=latents.dtype)
            chunk_size = latents.shape[1] // num_explanations
            for i in range(num_explanations):
                mask.data[i, indices[(i * chunk_size):((i + 1) * chunk_size)]] = 0
            mask = mask[None, ...]
        else:
            mask = torch.ones(1, num_explanations, latents.shape[1],
                              device=latents.device, dtype=latents.dtype)

        return mask


class Dive(ExplainerBase):
    """Main class to generate counterfactuals"""

    def __init__(self, exp_dict, data_path):
        """Constructor
        Args:
        exp_dict (dict): hyperparameter dictionary
        savedir (str): root path to experiment directory
        data_path (str): root path to datasets and pretrained models
        """
        super().__init__()
        self.exp_dict = exp_dict
        self.data_path = data_path
        self.generator_dict = self.exp_dict["generator_dict"]
        self.generator_path = self.generator_dict["weights"]
        self.classifier_dict = self.exp_dict["classifier_dict"]
        self.classifier_path = self.classifier_dict["weights"]
        # self.current_attribute = self.dataset.dataset.all_attributes.index(self.exp_dict['attribute'])
        self.classifier = self.load_classifier()
        self.generator = self.load_generator()
        self.oracle = self.generator.oracle
        self.appended_images = 0

        self.read_or_write_cache()
        # self.select_data_subset()

    def load_classifier(self):

        """Helper function to load the classifier model"""

        print("Loading classifier...")
        classifier = get_model(self.classifier_dict)
        classifier.load_state_dict(hu.torch_load(self.classifier_path))
        classifier.eval().cuda()
        return classifier


    def load_generator(self):

        """Helper function to load the generator model"""

        print("Loading generator...")
        generator = get_model(self.generator_dict)
        generator.load_state_dict(hu.torch_load(self.generator_path))
        generator.eval().cuda()
        return generator


    def get_loader(self, batch_size, mode):

        """Helper function to create a dataloader

            Args:
            batch_size (int): the batch_size

            Returns:
            torch.utils.data.DataLoader: dataloader without shuffling and chosen bs
        """
        dataset = getattr(self, f"{mode}_dataset")
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=False, shuffle=False)


    def read_or_write_cache(self):

        """Compute and store the Fisher Information Matrix
            for future use as well as other auxiliary data

            Raises:

            TimeoutError: if two processes collide, one will wait for the other to finish to avoid repeating operations.
            If the wait is too long, it raises an error.
        """
        self.digest = f"cache_{hu.hash_dict(self.classifier_dict)}_{hu.hash_dict(self.generator_dict)}"
        self.digest = os.path.join(self.data_path, self.digest)

        if self.exp_dict.get("force_cache", False):
            os.remove(self.digest)

        if not os.path.isfile(self.digest):
        # if not os.path.isfile(self.digest):
        #     try:
        #         os.makedirs(f"{self.digest}.lock")
        #         lock = False
        #     except FileExistsError:
        #         lock = True
        #     if lock:
        #         print(f"Waiting for another process to finish computing cache or delete {self.digest}.lock to continue")
        #         t = time.time()
        #         while os.path.isdir(f"{self.digest}.lock"):
        #             if (time.time() - t) / 3600 > 2:
        #                 raise TimeoutError(f"Timout while waiting for another process to \
        #                                     finish computing cache on {self.digest}. Delete \
        #                                     if that is not the case")
        #             time.sleep(1)
        #         lock = False
            # self.write_cache()
            print("Caching FIM")
            self.write_cache("train")
            print("Caching latents")
            self.write_cache("val")

        self.read_cache()


    def select_data_subset(self):

        """Instead of using the whole dataset, we use a balanced set of correctly and incorrectly classified samples"""
        if self.exp_dict["n_samples"] > 0:
            preds = torch.sigmoid(self.logits).numpy()
            labels = self.val_dataset.dataset.y.astype(float)
            indices = []
            for confidence in [-0.9, -0.6, -0.4, -0.1, 0.1, 0.4, 0.6, 0.9]:
                # obtain samples that are closest to the required level of confidence
                indices.append(np.abs(labels - preds.max(1) - confidence).argsort()
                                [:self.exp_dict["n_samples"]])
            indices = np.concatenate(indices, 0)
            self.val_dataset.indices = indices


    def write_cache(self, mode):

        """Loops through the data and stores latents and FIM"""

        loader = self.get_loader(self.exp_dict['cache_batch_size'], mode)
        preds = []
        fishers = 0
        mus = []
        for idx, x, y, categorical_att, continuous_att in tqdm(loader):
            with torch.no_grad():
                x = x.cuda()
                labels = y.cuda()
                categorical_att = categorical_att.cuda()
                continuous_att = continuous_att.cuda()
                z = self.generator.model.embed_attributes(categorical_att, continuous_att)
                mus.append(z.cpu().numpy())
                # outputs = self.model.model.encode(x)
                # mu = outputs["z"]
                # mus.append(mu)
                # logvars.append(logvar.cpu())
                # logvars.append(0)
                # b, c = mu.size()
                # z = mu.data.clone()
                self.first = True

            def jacobian_forward(z):

                reconstruction = self.generator.model.decode(z)
                logits = self.classifier.model(reconstruction)
                if self.first:
                    preds.append(logits.data.cpu().numpy())
                    self.first = False
                y = torch.distributions.Bernoulli(logits=logits).sample().detach()
                logits = logits * y + (1 - logits) * (1 - y)
                loss = logits.sum(0)

                return loss

            grads = torch.autograd.functional.jacobian(jacobian_forward, z)
            if mode == "train":
                b, c = z.size()
                num_classes = self.exp_dict["dataset"]["num_classes"]

                with torch.no_grad():
                    fisher = torch.matmul(grads[:, :, :, None], grads[:, :, None, :]).view(num_classes, b, c, c).sum(1).cpu()
                fishers += fisher.numpy()
                del(fisher)
                to_save = dict(fishers=fishers)
            elif mode == "val":
                to_save = dict(mus=np.concatenate(mus, 0),
                               logits=np.concatenate(preds, 0))
            del(z)

        with h5py.File(self.digest, 'a') as outfile:
            for k, v in to_save.items():
                outfile[k] = v
                # os.removedirs(f"{self.digest}.lock")
                print("Done.")


    def read_cache(self):

        """Reads cached data from disk"""
        print("Loading from %s" % self.digest)
        self.loaded_data = h5py.File(self.digest, 'r')
        try:
            self.fisher = torch.from_numpy(self.loaded_data['fishers'][...])
            self.logits = torch.from_numpy(self.loaded_data['logits'][...])
            self.mus = torch.from_numpy(self.loaded_data['mus'][...])
        except:
            print("FIM not found in the hdf5 cache")
            print("Done.")


    def attack_batch(self, batch):

        """Uses gradient descent to compute counterfactual explanations
        Args:
        batch (tuple): a batch of image ids, images, and labels
        Returns:
            dict: a dictionary containing the whole attack history
        """
        idx, images, labels, categorical_att, continuous_att = batch
        idx = idx.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()

        predicted_labels = torch.sigmoid(self.logits[idx])

        predicted_labels = predicted_labels.float().cuda()

        diversity_weight = self.exp_dict['diversity_weight']
        lasso_weight = self.exp_dict['lasso_weight']
        reconstruction_weight = self.exp_dict['reconstruction_weight']

        # import matplotlib.pyplot as plt
        # lambda_range=np.linspace(0,1,10)
        # fig, axs = plt.subplots(2,5, figsize=(15, 6))
        # fig.subplots_adjust(hspace = .5, wspace=.001)
        # axs = axs.ravel()
        # att_1 = categorical_att[0].unsqueeze(0)
        # att_1[:, 1] = np.random.choice(list(range(0, 1072)))
        # att_2 = att_1.clone()
        # att_2[:, 1] = np.random.choice(list(range(0, 1072)))
        # import matplotlib
        # matplotlib.use("TKAgg")
        # for ind, l in enumerate(lambda_range):
        #     latent_1 = self.generator.model.embed_attributes(att_1, continuous_att[0].unsqueeze(0))
        #     latent_2 = self.generator.model.embed_attributes(att_2, continuous_att[0].unsqueeze(0))
        #     inter_latent = latent_1 * l + (1 - l) * latent_2
        #     inter_image = self.generator.model.decode(inter_latent)
        #     image = inter_image.clamp(0, 1).view(3, 32, 32).permute(1, 2, 0).cpu().detach().numpy()
        #     axs[ind].imshow(image, cmap='gray')
        #     axs[ind].set_title('lambda_val='+str(round(l,1)))
        # plt.show()
        latents = self.mus[idx].cuda()
        b, c = latents.size()

        num_explanations = self.exp_dict['num_explanations']
        epsilon = torch.randn(b, num_explanations, c, requires_grad=True, device=latents.device)
        epsilon.data *= 0.01

        mask = self.get_mask(batch, latents)

        optimizer = torch.optim.Adam([epsilon], lr=self.exp_dict['lr'], weight_decay=0)

        attack_history = []

        class DecoderClassifier(torch.nn.Module):
            def __init__(self, g, c):
                super().__init__()
                self.g = g
                self.c = c

            def forward(self, x):
                decoded = self.g(x)
                return decoded, self.c(decoded)

        decoder_classifier = DecoderClassifier(self.generator.model.decoder, self.classifier.model)

        for it in range(self.exp_dict['max_iters']):
            optimizer.zero_grad()
            div_regularizer = 0
            lasso_regularizer = 0
            reconstruction_regularizer = 0

            repeat_dim = epsilon.size(0) // mask.size(0)
            epsilon.data = epsilon.data * mask.repeat(repeat_dim, 1, 1)
            # epsilon.data = epsilon.data * mask
            z_perturbed = latents[:, None, :].detach() + epsilon
            if diversity_weight > 0:
                epsilon_normed = epsilon
                epsilon_normed = F.normalize(epsilon_normed, 2, -1)
                div_regularizer = torch.matmul(epsilon_normed, epsilon_normed.permute(0, 2, 1))
                div_regularizer = div_regularizer * (1 - torch.eye(div_regularizer.shape[-1],
                                                                    dtype=div_regularizer.dtype,
                                                                    device=div_regularizer.device))[None, ...]
                div_regularizer = (div_regularizer ** 2).sum()

            decoded, logits = decoder_classifier(z_perturbed.view(b * num_explanations, c))
            bn, ch, h, w = decoded.size()
            if reconstruction_weight > 0:
                reconstruction_regularizer = torch.abs(images[:, None, ...] - decoded.view(b, num_explanations, ch, h, w)).sum()

            lasso_regularizer = torch.abs(z_perturbed - latents[:, None, :]).sum()

            regularizer = lasso_regularizer * lasso_weight + \
                                    div_regularizer * diversity_weight + \
                                    reconstruction_regularizer * reconstruction_weight

            # regularizer = regularizer / mask.expand_as(z_perturbed).sum()
            regularizer = regularizer / mask.repeat(repeat_dim, 1, 1).sum()

            # loss_attack = F.binary_cross_entropy_with_logits(logits, 1 - predicted_labels.repeat(num_explanations, 1), reduction='none')
            loss_attack = F.binary_cross_entropy_with_logits(logits, 1 - predicted_labels.repeat(num_explanations, 1))
            # loss_attack = (loss_attack.view(b, num_explanations) * opt_mask).mean()
            loss = loss_attack + regularizer
            loss.backward()
            optimizer.step()
            attack_history = dict(
                                iter=np.array([it]),
                                idx=idx.data.cpu().numpy(),
                                logits=logits.data.cpu().view(b, num_explanations, -1).numpy(),
                                labels=np.array(labels.squeeze().data.cpu().numpy()),
                                loss=np.array([float(loss)]),
                                loss_attack=np.array([float(loss_attack)]),
                                reconstruction_regularizer=np.array([float(reconstruction_regularizer)]),
                                div_regularizer=np.array([float(div_regularizer)]),
                                lasso_regularizer=np.array([float(lasso_regularizer)]),
                                )  # instead of append, directly set dict due to memory constraints
            success_rate = float((logits.argmax(1) != predicted_labels.repeat(num_explanations, 1).argmax(1)).float().mean())
            attack_history['success_rate'] = np.array([success_rate])
            if success_rate >= self.exp_dict["stop_batch_threshold"]:
                break

        pred_dict = {}
        with torch.no_grad():
            torch.cuda.empty_cache()
            y_perturbed = self.classifier.model(decoded)
            oracle_preds_attack = self.oracle(decoded)

            pred_dict["font"] = oracle_preds_attack["pred_font"].argmax(1).view(b, num_explanations).cpu().tolist()
            pred_dict["char"] = oracle_preds_attack["pred_char"].argmax(1).view(b, num_explanations).cpu().tolist()
            # set continous attributes dict
            for k, v in idx_to_name.items():
                pred_dict[v] = oracle_preds_attack["pred_continuous"][:, k].view(b, num_explanations).cpu().tolist()

            attack_history["oracle_preds_attack"] = json.dumps(pred_dict)

            pred_dict["font"] = categorical_att[:, -1].cpu().tolist()
            pred_dict["char"] = categorical_att[:, -2].cpu().tolist()

            for k, v in idx_to_name.items():
                pred_dict[v] = continuous_att[:, k].cpu().tolist()

            attack_history["oracle_preds_reconstruction"] = json.dumps(pred_dict)

            attack_history["classifier_preds_attack"] = y_perturbed.cpu().numpy()
            attack_history["attacked_latents"] = z_perturbed.cpu().numpy()
            attack_history['latent_similarity'] = (F.normalize(latents, 2, -1).repeat(num_explanations, 1).view(b, num_explanations, -1) *
                                                    F.normalize(z_perturbed, 2, -1).view(b, num_explanations, -1)).sum(-1).view(b, -1).cpu().numpy()
            if self.appended_images < 5:
                attack_history["decoded"] = decoded.view(b, num_explanations, *images.size()[1:]).cpu().numpy()
                attack_history["images"] = images.cpu().numpy()
                self.appended_images += 1

        return attack_history


    def attack_dataset(self):
        """Loops over the dataset and generates counterfactuals for all the samples"""

        loader = self.get_loader(self.exp_dict['batch_size'] // self.exp_dict['num_explanations'], mode="val")
        attack_histories = {}
        for batch in tqdm(loader):
            history = self.attack_batch(batch)
            for k, v in history.items():
                if k in attack_histories:
                    attack_histories[k].append(v)
                else:
                    attack_histories[k] = [v]
            # if self.appended_images == 5:
            #     break
            # print(f"total_loss: {history['loss']},",
            #                     f"loss_attack: {history['loss_attack']},",
            #                     f"reconstruction: {history['reconstruction_regularizer']},",
            #                     f"lasso: {history['lasso_regularizer']},",
            #                     f"diversity: {history['div_regularizer']},",
            #                     f"success_rate: {history['success_rate']}")

        return attack_histories


    def get_mask(self, batch, latents):
        """Helper function that outputs a binary mask for the latent
            space during the counterfactual explanation
        Args:
            latents (torch.Tensor): dataset latents (precomputed)
        Returns:
            torch.Tensor: latents mask
        """
        method = self.exp_dict["method"]
        num_explanations = self.exp_dict['num_explanations']

        if 'fisher' in method:
            if self.exp_dict['fisher_samples'] <= 0:
                # fishers = [self.fisher[self.current_attribute]]
                fishers = self.fisher
            else:
                fishers = self.get_pointwise_fisher(batch, self.exp_dict['fisher_samples'])

        if method in ["fisher_chunk"]:
            masks = []
            for fisher in fishers:
                indices = torch.diagonal(fisher).argsort(descending=True, dim=-1)
                mask = torch.ones(num_explanations, latents.shape[1], device=latents.device, dtype=latents.dtype)
                chunk_size = latents.shape[1] // num_explanations
                for i in range(num_explanations):
                    mask.data[i, indices[(i * chunk_size) :((i + 1) * chunk_size)]] = 0
                masks.append(mask)
            mask = torch.stack(masks, 0)
        elif method in ["fisher_range"]:
            masks = []
            for fisher in fishers:
                indices = torch.diagonal(fisher).argsort(descending=True, dim=-1)
                mask = torch.ones(num_explanations, latents.shape[1], device=latents.device, dtype=latents.dtype)
                for i in range(num_explanations):
                    mask.data[i, indices[0:i]] = 0
                masks.append(mask)
            mask = torch.stack(masks, 0)
        elif method in ["fisher_spectral", "fisher_spectral_inv"]:
            masks = []
            for fisher in fishers:
                scluster = SpectralClustering(n_clusters=num_explanations, affinity='precomputed',
                                                assign_labels='discretize', random_state=0, eigen_solver='arpack', eigen_tol=1e-6)
                affinity = fisher.numpy()
                affinity = affinity - affinity.min()
                affinity /= affinity.max()
                scluster.fit(affinity)
                mask = torch.zeros(num_explanations, latents.shape[1], device=latents.device)
                for i in range(num_explanations):
                    mask[i, torch.from_numpy(scluster.labels_).to(latents.device) == i] = 1
                masks.append(mask)
            mask = torch.stack(masks, 0)
            if 'inv' in method:
                mask = 1 - mask
        elif method in ["random"]:
            indices = torch.randperm(latents.shape[1], device=latents.device)
            mask = torch.ones(num_explanations, latents.shape[1], device=latents.device, dtype=latents.dtype)
            chunk_size = latents.shape[1] // num_explanations
            for i in range(num_explanations):
                mask.data[i, indices[(i * chunk_size):((i + 1) * chunk_size)]] = 0
            mask = mask[None, ...]
        else:
            mask = torch.ones(1, num_explanations, latents.shape[1],
                              device=latents.device, dtype=latents.dtype)

        return mask
