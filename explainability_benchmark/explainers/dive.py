import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
from .base import ExplainerBase, LatentExplainerBase


torch.backends.cudnn.benchmark = True


class Dive(ExplainerBase):
    """Main class to generate counterfactuals"""

    def __init__(self, encoder, generator, classifier, train_loader, num_explanations=8,
                 diversity_weight=1, data_path="data", lr=0.01, lasso_weight=10, reconstruction_weight=5, max_iters=20,
                 method="fisher_spectral"):

        """Constructor
        Args:
        exp_dict (dict): hyperparameter dictionary
        savedir (str): root path to experiment directory
        data_path (str): root path to datasets and pretrained models
        """

        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.generator = generator
        self.lr = lr
        self.train_loader = train_loader
        self.diversity_weight = diversity_weight
        self.lasso_weight = lasso_weight
        self.reconstruction_weight = reconstruction_weight
        self.max_iters = max_iters
        self.num_explanations = num_explanations
        self.method = method
        self.cache = False


    def read_or_write_fim(self):

        if "fishers" not in self.loaded_data:
            print("Caching FIM")
            self.write_fim()
        self.read_fim()


    def write_fim(self):

        for idx, x, y, categorical_att, continuous_att in tqdm(self.train_loader):
            x = x.cuda()
            categorical_att = categorical_att.cuda()
            continuous_att = continuous_att.cuda()
            z = self.generator.model.embed_attributes(categorical_att, continuous_att)

            def jacobian_forward(z):

                reconstruction = self.generator.model.decode(z)
                logits = self.classifier.model(reconstruction)
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


    def read_fim(self):
        print("Reading FIM...")
        self.fisher = torch.from_numpy(self.loaded_data['fishers'][...])
        print("Done...")


    def explain_batch(self, latents, logits, images, classifier, generator):

        """Uses gradient descent to compute counterfactual explanations
        Args:
        batch (tuple): a batch of image ids, images, and labels
        Returns:
            dict: a dictionary containing the whole attack history
        """

        # # TODO this is hacky
        if not self.cache:
            self.read_or_write_fim()
            self.cache = True

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
        b, c = latents.size()

        predicted_labels = torch.sigmoid(logits)

        predicted_labels = predicted_labels.float().cuda()


        num_explanations = self.num_explanations
        epsilon = torch.randn(b, num_explanations, c, requires_grad=True, device=latents.device)
        epsilon.data *= 0.01
        # epsilon.data *= 0

        mask = self.get_mask(latents)

        optimizer = torch.optim.Adam([epsilon], lr=self.lr, weight_decay=0)

        for _ in range(self.max_iters):
            optimizer.zero_grad()
            regularizers = []

            repeat_dim = epsilon.size(0) // mask.size(0)
            epsilon.data = epsilon.data * mask.repeat(repeat_dim, 1, 1)
            # epsilon.data = epsilon.data * mask
            z_perturbed = latents[:, None, :].detach() + epsilon

            decoded = generator(z_perturbed.view(b * num_explanations, c))
            logits = classifier(decoded)
            _, ch, h, w = decoded.size()
            decoded = decoded.view(b, num_explanations, ch, h, w)

            if self.diversity_weight > 0:
                regularizers.append(self.compute_div_regularizer(epsilon))

            if self.reconstruction_weight > 0:
                regularizers.append(self.compute_rec_regularizer(images, decoded))

            if self.lasso_weight > 0:
                regularizers.append(self.compute_lasso_regularizer(z_perturbed, latents))


            regularizer = sum(regularizers)

            regularizer = regularizer / mask.repeat(repeat_dim, 1, 1).sum()
            loss = self.compute_loss(logits, predicted_labels.repeat(num_explanations, 1), regularizer)

            loss.backward()
            optimizer.step()

        return z_perturbed


    def compute_loss(self, logits, predicted_labels, regularizer):

        loss_attack = F.binary_cross_entropy_with_logits(logits, 1 - predicted_labels)
        loss = loss_attack + regularizer

        return loss


    def compute_lasso_regularizer(self, z_perturbed, latents):
        latents = latents[:, None, :]
        # TODO temporary
        lasso_c = torch.abs(z_perturbed[..., :128] - latents[..., :128]).mean()
        lasso_f = torch.abs(z_perturbed[..., 128:256] - latents[..., 128:256]).mean()
        lasso_cont = torch.abs(z_perturbed[..., 256:] - latents[..., 256:]).mean()
        lasso_regularizer = lasso_cont + lasso_f + lasso_c
        return (lasso_regularizer * self.lasso_weight).item()


    def compute_rec_regularizer(self, images, decoded):
        reconstruction_regularizer = torch.abs(images[:, None, ...] - decoded).sum()

        return (reconstruction_regularizer * self.reconstruction_weight).item()


    def compute_div_regularizer(self, epsilon):

        epsilon_normed = epsilon
        epsilon_normed = F.normalize(epsilon_normed, 2, -1)
        div_regularizer = torch.matmul(epsilon_normed, epsilon_normed.permute(0, 2, 1))
        div_regularizer = div_regularizer * (1 - torch.eye(div_regularizer.shape[-1],
                                                            dtype=div_regularizer.dtype,
                                                            device=div_regularizer.device))[None, ...]
        div_regularizer = (div_regularizer ** 2).sum()

        return (div_regularizer * self.diversity_weight).item()


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
