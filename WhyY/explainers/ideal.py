import torch
import numpy as np
from .base import ExplainerBase

class IdealExplainer(ExplainerBase):

    def __init__(self, encoder, generator, dataset, num_explanations=10):

        super().__init__()
        self.num_explanations = num_explanations
        self.encoder = encoder
        self.generator = generator
        # dataset.dataset.dataset (the world is not ready for this)
        self.correlated_att = dataset.dataset.dataset.correlated


    def explain_batch(self, latents, logits, images, classifier, generator):

        mean = self.latent_mean[3:6]
        std = self.latent_std[3:6]
        b, c = latents.size()

        targets = 1 - logits.argmax(1)
        n_corr = len(self.correlated_att) // 2
        correlation = (self.correlated_att[:n_corr], self.correlated_att[n_corr:])
        z = latents[:, None, ].repeat(1, self.num_explanations, 1).clone()

        weights_font = self.generator.model.font_embedding.weight

        for i, t in enumerate(targets):
            t = int(t)
            font = np.random.choice(correlation[t], self.num_explanations, replace=True)
            z[i, :, 3:6] = (weights_font[font] - mean.cuda()) / std.cuda()

        z_perturbed = z

        return z_perturbed.view(b, self.num_explanations, c)
