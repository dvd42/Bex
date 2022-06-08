import torch
import numpy as np
from tqdm import tqdm

from .base import ExplainerBase


class LCF(ExplainerBase):

    """ Latent-CF explainer as described in https://arxiv.org/abs/2012.09301

    Args:
        num_explanations (``int``, optional): number of counterfactuals to be generated (default: 10)
        lr (``float``, optional): learning rate (default: 0.1)
        num_iters (``int``, optional): max number of gradient descent steps to perform without convergence (default: 50)
        p (``float``, optional): probability :math:`p` of target counterfactual class.
        tolerance (``float``, optional) tolerance :math:`tol` value for the loss function (default: 0.5)
    """

    def __init__(self, num_explanations=10, lr=0.1, num_iters=50, p=0.1, tolerance=0.5):

        super().__init__()

        self.num_explanations = num_explanations
        self.tolerance = tolerance
        self.p = p
        self.lr = lr
        self.num_iters = num_iters


    def explain_batch(self, latents, logits, images, classifier, generator):

        b, c = latents.size()
        labels = logits.argmax(1)
        loss = float("inf")

        # initialize set of explanations close to the original examples (same as DICE)
        labels = labels[:, None].repeat(1, self.num_explanations).view(-1)
        z_perturbed = latents[:, None, :].repeat(1, self.num_explanations, 1).view(b, -1, c)
        for i in range(1, self.num_explanations):
            z_perturbed[:, i] = z_perturbed[:, i] + 0.01 * i

        i = 0
        z_perturbed = z_perturbed.view(-1, c)
        z_perturbed.requires_grad = True
        while loss > self.tolerance and i < self.num_iters:

            p_logits = classifier(generator(z_perturbed))
            diff = torch.sigmoid(p_logits)[torch.arange(labels.shape[0]), labels] - self.p
            loss = diff.abs().mean()
            loss.backward()

            z_perturbed = z_perturbed - self.lr * z_perturbed.grad
            z_perturbed.retain_grad()
            i += 1

        return z_perturbed.view(b, -1, c)
