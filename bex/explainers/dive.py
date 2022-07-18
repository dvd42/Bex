import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
from .base import ExplainerBase


class Dive(ExplainerBase):

    """
    DiVE algorithm as described in https://arxiv.org/abs/2103.10226

    Args:
        num_explanations (``int``, optional): number of counterfactuals to be generated (default: 10)
        lr (``float``, optional): learning rate (default: 0.1)
        num_iters (``int``, optional): number of gradient descent steps to perform (default: 50)
        diversity_weight (``float``, optional): weight of the diversity term in the loss function (default: 0)
        lasso_weight (``float``, optional): factor :math:`\\gamma` that controls the sparsity of the latent space (default: 0.1)
        reconstruction_weight (``float``, optional): weight of the reconstruction term in the loss function (default: 0.001)
        method (``string``, optional): method used for gradient masking (default: `'fisher_spectral'`)
    """

    def __init__(self, num_explanations=10, lr=0.1, num_iters=50,
                 diversity_weight=0.001, lasso_weight=0.1, reconstruction_weight=0.0001,
                 method="fisher_spectral"):


        super().__init__()
        self.lr = lr
        self.diversity_weight = diversity_weight
        self.lasso_weight = lasso_weight
        self.reconstruction_weight = reconstruction_weight
        self.num_iters = num_iters
        self.num_explanations = num_explanations
        self.method = method
        self.cache = False


    def __read_or_write_fim(self, generator, classifier):

        if "fishers" not in self.loaded_data:
            print("Caching FIM")
            self.__write_fim(generator, classifier)
        self.__read_fim()


    def __write_fim(self, generator, classifier):

        train_mus = (self.train_mus.cuda() - self.latent_mean.cuda()) / self.latent_std.cuda()
        for z in tqdm(train_mus.chunk(train_mus.shape[0] // 512)):


            def jacobian_forward(z):

                reconstruction = generator(z)
                logits = classifier(reconstruction)
                y = torch.distributions.Bernoulli(logits=logits).sample().detach()
                logits = logits * y + (1 - logits) * (1 - y)
                loss = logits.sum(0)

                return loss

            grads = torch.autograd.functional.jacobian(jacobian_forward, z)
            b, c = z.size()
            fishers = 0

            with torch.no_grad():
                fisher = torch.matmul(grads[:, :, :, None], grads[:, :, None, :]).view(self.num_classes, b, c, c).sum(1).cpu()
            fishers += fisher.numpy()
            del fisher
            del z
        to_save = dict(fishers=fishers)

        for k, v in to_save.items():
            self.loaded_data[k] = v
            print("Done.")


    def __read_fim(self):
        print("Reading FIM...")
        self.fisher = torch.from_numpy(self.loaded_data['fishers'][...])
        print("Done...")


    def explain_batch(self, latents, logits, images, classifier, generator):


        # TODO this is hacky
        if not self.cache:
            self.__read_or_write_fim(generator, classifier)
            self.cache = True

        b, c = latents.size()

        predicted_labels = torch.sigmoid(logits)

        predicted_labels = predicted_labels.float().cuda()


        num_explanations = self.num_explanations
        epsilon = torch.randn(b, num_explanations, c, requires_grad=True, device=latents.device)
        epsilon.data *= 0.01
        # epsilon.data *= 0

        mask = self.__get_mask(latents)

        optimizer = torch.optim.Adam([epsilon], lr=self.lr, weight_decay=0)

        predicted_labels = predicted_labels[:, None, :].repeat(1, num_explanations, 1).view(-1, logits.shape[1])
        for _ in range(self.num_iters):
            optimizer.zero_grad()
            regularizer = 0

            repeat_dim = epsilon.size(0) // mask.size(0)
            epsilon.data = epsilon.data * mask.repeat(repeat_dim, 1, 1)
            # epsilon.data = epsilon.data * mask
            z_perturbed = latents[:, None, :].detach() + epsilon

            decoded = generator(z_perturbed.view(b * num_explanations, c))
            logits = classifier(decoded)
            _, ch, h, w = decoded.size()
            decoded = decoded.view(b, num_explanations, ch, h, w)

            if self.diversity_weight > 0:
                regularizer += self.__compute_div_regularizer(epsilon)

            if self.reconstruction_weight > 0:
                regularizer += self.__compute_rec_regularizer(images, decoded)

            if self.lasso_weight > 0:
                regularizer += self.__compute_lasso_regularizer(z_perturbed, latents)


            regularizer = regularizer / mask.repeat(repeat_dim, 1, 1).sum()
            loss = self.__compute_loss(logits, predicted_labels, regularizer)

            loss.backward()
            optimizer.step()

        return z_perturbed


    def __compute_loss(self, logits, predicted_labels, regularizer):

        loss_attack = F.binary_cross_entropy_with_logits(logits, 1 - predicted_labels)
        loss = loss_attack + regularizer

        return loss


    def __compute_lasso_regularizer(self, z_perturbed, latents):
        latents = latents[:, None, :]
        lasso_regularizer = torch.abs(z_perturbed - latents).sum()
        return lasso_regularizer * self.lasso_weight


    def __compute_rec_regularizer(self, images, decoded):
        reconstruction_regularizer = torch.abs(images[:, None, ...] - decoded).sum()

        return reconstruction_regularizer * self.reconstruction_weight


    def __compute_div_regularizer(self, epsilon):

        epsilon_normed = epsilon
        epsilon_normed = F.normalize(epsilon_normed, 2, -1)
        div_regularizer = torch.matmul(epsilon_normed, epsilon_normed.permute(0, 2, 1))
        div_regularizer = div_regularizer * (1 - torch.eye(div_regularizer.shape[-1],
                                                            dtype=div_regularizer.dtype,
                                                            device=div_regularizer.device))[None, ...]
        div_regularizer = (div_regularizer ** 2).sum()

        return div_regularizer * self.diversity_weight


    def __get_mask(self, latents):

        method = self.method
        num_explanations = self.num_explanations

        if 'fisher' in method:
            fishers = self.fisher

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
