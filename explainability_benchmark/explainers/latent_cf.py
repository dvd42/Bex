import torch
import numpy as np
from tqdm import tqdm

from .base import ExplainerBase, LatentExplainerBase


class LCF(ExplainerBase):

    def __init__(self, num_explanations=8, p=0.1, tolerance=0.1, lr=0.05, max_iters=50):

        super().__init__()

        self.num_explanations = num_explanations
        self.tolerance = tolerance
        self.p = p
        self.lr = lr
        self.max_iters = max_iters


    def explain_batch(self, latents, logits, images, classifier, generator):

        b, c = latents.size()
        labels = logits.argmax(1)
        loss = float("inf")

        # initialize set of explanations close to the original examples (same as DICE)
        labels = labels.repeat(self.num_explanations).view(-1)
        z_perturbed = latents[:, None, :].repeat(1, self.num_explanations, 1).view(b, -1, c)
        for i in range(1, self.num_explanations):
            z_perturbed[:, i] = z_perturbed[:, i] + 0.01 * i

        i = 0
        z_perturbed = z_perturbed.view(-1, c)
        z_perturbed.requires_grad = True
        while loss > self.tolerance and i < self.max_iters:

            p_logits = classifier(generator(z_perturbed))
            diff = torch.sigmoid(p_logits)[torch.arange(labels.shape[0]), labels] - self.p
            loss = diff.abs().mean()
            loss.backward()

            z_perturbed = z_perturbed - self.lr * z_perturbed.grad
            z_perturbed.retain_grad()
            i += 1

        return z_perturbed.view(b, -1, c)


class LatentLCF(LatentExplainerBase):

    def __init__(self, num_explanations=8, p=0.1, tolerance=0.1, lr=0.05, max_iters=500):

        super().__init__()

        self.num_explanations = num_explanations
        self.tolerance = tolerance
        self.p = p
        self.lr = lr
        self.max_iters = max_iters


    def explain_batch(self, latents, logits, classifier):

        b, c = latents.size()
        labels = logits.argmax(1)
        loss = float("inf")

        # initialize set of explanations close to the original examples (same as DICE)
        labels = labels.repeat(self.num_explanations).view(-1)
        z_perturbed = latents.repeat(self.num_explanations, 1).view(b, -1, c)
        for i in range(1, self.num_explanations):
            z_perturbed[:, i] = z_perturbed[:, i] + 0.01 * i

        i = 0
        z_perturbed = z_perturbed.view(-1, c)
        z_perturbed.requires_grad = True
        while loss > self.tolerance and i < self.max_iters:

            p_logits = classifier(z_perturbed)
            diff = torch.sigmoid(p_logits)[:, labels][:, 0] - self.p
            loss = diff.abs().mean()
            loss.backward()

            z_perturbed = z_perturbed - self.lr * z_perturbed.grad
            z_perturbed.retain_grad()
            i += 1

        return z_perturbed
