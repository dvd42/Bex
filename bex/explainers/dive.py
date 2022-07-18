import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
from .base import ExplainerBase


class Dive(ExplainerBase):
    """Main class to generate counterfactuals"""

    def __init__(self, num_explanations=10,
                 diversity_weight=0, lr=0.1, lasso_weight=0.1, reconstruction_weight=0.001, max_iters=50, sparsity_weight=0., beta=0.1,
                 method="fisher_spectral"):

        """Constructor
        Args:
        exp_dict (dict): hyperparameter dictionary
        savedir (str): root path to experiment directory
        """

        super().__init__()
        self.lr = lr
        self.diversity_weight = diversity_weight
        self.lasso_weight = lasso_weight
        self.reconstruction_weight = reconstruction_weight
        self.sparsity_weight = sparsity_weight
        self.max_iters = max_iters
        self.beta = beta
        self.num_explanations = num_explanations
        self.method = method
        self.cache = False


    def read_or_write_fim(self, generator, classifier):

        if "fishers" not in self.loaded_data:
            print("Caching FIM")
            self.write_fim(generator, classifier)
        self.read_fim()


    def write_fim(self, generator, classifier):

        # for idx, x, y, categorical_att, continuous_att in tqdm(self.train_loader):
        #     x = x.cuda()
        #     categorical_att = categorical_att.cuda()
        #     continuous_att = continuous_att.cuda()
        #     z = self.generator.model.embed_attributes(categorical_att, continuous_att)

            # self.train_mus = torch.from_numpy(self.loaded_data['train_mus'][...])
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
        to_save = dict(fishers_norm=fishers)

        for k, v in to_save.items():
            self.loaded_data[k] = v
            print("Done.")


    def read_fim(self):
        print("Reading FIM...")
        self.fisher = torch.from_numpy(self.loaded_data['fishers'][...])
        # self.fisher = torch.from_numpy(self.loaded_data['fishers_norm'][...])
        print("Done...")


    def explain_batch(self, latents, logits, images, classifier, generator):

        """Uses gradient descent to compute counterfactual explanations
        Args:
        batch (tuple): a batch of image ids, images, and labels
        Returns:
            dict: a dictionary containing the whole attack history
        """

        # TODO this is hacky
        if not self.cache:
            self.read_or_write_fim(generator, classifier)
            self.cache = True

        b, c = latents.size()

        predicted_labels = torch.sigmoid(logits)

        predicted_labels = predicted_labels.float().cuda()


        num_explanations = self.num_explanations
        epsilon = torch.randn(b, num_explanations, c, requires_grad=True, device=latents.device)
        epsilon.data *= 0.01
        # epsilon.data *= 0

        mask = self.get_mask(latents)

        optimizer = torch.optim.Adam([epsilon], lr=self.lr, weight_decay=0)

        predicted_labels = predicted_labels[:, None, :].repeat(1, num_explanations, 1).view(-1, logits.shape[1])
        for _ in range(self.max_iters):
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
                regularizer += self.compute_div_regularizer(epsilon)

            if self.reconstruction_weight > 0:
                regularizer += self.compute_rec_regularizer(images, decoded)

            if self.lasso_weight > 0:
                regularizer += self.compute_lasso_regularizer(z_perturbed, latents)

            if self.sparsity_weight > 0:
                regularizer += self.compute_sparsity_regularizer(z_perturbed, latents)


            regularizer = regularizer / mask.repeat(repeat_dim, 1, 1).sum()
            loss = self.compute_loss(logits, predicted_labels, regularizer)

            loss.backward()
            optimizer.step()

        return z_perturbed


    def compute_loss(self, logits, predicted_labels, regularizer):

        loss_attack = F.binary_cross_entropy_with_logits(logits, 1 - predicted_labels)
        loss = loss_attack + regularizer

        return loss

    
    def compute_sparsity_regularizer(self, z_perturbed, latents):

        b, ne, c = z_perturbed.size()
        latents = latents[:, None, :].repeat(1, ne, 1)
        z_perturbed = self._cosine_embedding(z_perturbed).view(b * ne, -1)
        perturbations = z_perturbed.clone()
        latents = self._cosine_embedding(latents, binarize=True).view(b * ne, -1)
        perturbations[:, -5:] = (latents[:, -5:] - perturbations[:, -5:]).abs()
        perturbations[:, :-5] = (perturbations[:, :-5] * (1 - latents[:, :-5])).abs()

        p_beta = torch.distributions.exponential.Exponential(rate=1/self.beta).rsample((perturbations.size())).cuda()
        beta_hat = perturbations / perturbations.shape[0] + 1e-6
        
        regularizer = torch.where(beta_hat > p_beta, 1/beta_hat - p_beta/beta_hat**2, torch.zeros_like(beta_hat))
        beta_hat = latents / latents.shape[0] + 1e-6
        regularizer2 = torch.where(beta_hat > p_beta, 1/beta_hat - p_beta/beta_hat**2, torch.zeros_like(beta_hat))
        print("real", (regularizer.mean() * self.sparsity_weight).item())
        print("ideal", (regularizer2.mean() * self.sparsity_weight).item())

        # skl = torch.where(p_beta > self.beta, (p_beta.log() + perturbations/p_beta - perturbations.log() - 1.), torch.zeros_like(p_beta))
        return regularizer.mean() * self.sparsity_weight


    def compute_lasso_regularizer(self, z_perturbed, latents):
        latents = latents[:, None, :]
        lasso_regularizer = torch.abs(z_perturbed - latents).sum()
        return lasso_regularizer * self.lasso_weight


    def compute_rec_regularizer(self, images, decoded):
        reconstruction_regularizer = torch.abs(images[:, None, ...] - decoded).sum()

        return reconstruction_regularizer * self.reconstruction_weight


    def compute_div_regularizer(self, epsilon):

        epsilon_normed = epsilon
        epsilon_normed = F.normalize(epsilon_normed, 2, -1)
        div_regularizer = torch.matmul(epsilon_normed, epsilon_normed.permute(0, 2, 1))
        div_regularizer = div_regularizer * (1 - torch.eye(div_regularizer.shape[-1],
                                                            dtype=div_regularizer.dtype,
                                                            device=div_regularizer.device))[None, ...]
        div_regularizer = (div_regularizer ** 2).sum()

        return div_regularizer * self.diversity_weight


    def get_mask(self, latents):
        """Helper function that outputs a binary mask for the latent
            space during the counterfactual explanation
        Args:
            latents (torch.Tensor): dataset latents (precomputed)
        Returns:
            torch.Tensor: latents mask
        """
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

    def _cosine_embedding(self, z, binarize=False):

        b, ne, c = z.size()
        z = z.view(-1, c)
        weights_char = self.generator.model.char_embedding.weight.clone()
        weights_font = self.generator.model.font_embedding.weight.clone()
        weights_char = (weights_char - self.latent_mean[:3].cuda()) / self.latent_std[:3].cuda()
        weights_font = (weights_font - self.latent_mean[3:6].cuda()) / self.latent_std[3:6].cuda()
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
