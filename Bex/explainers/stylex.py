import torch
import numpy as np
from tqdm import tqdm

from .base import ExplainerBase

torch.backends.cudnn.deterministic = True


class Stylex(ExplainerBase):

    """ StylEx explainer as described in https://arxiv.org/abs/2104.13369

    num_explanations (``int``, optional): number of counterfactuals to be generated (default: 10)
    t (``float``, optional): perturbation threshold :math:`t` to consider a sample *explained* (default: 0.3)
    shift_size (``float``, optional): amount of shift applied to each coordinate (default: 0.8)
    strategy(``string``, optional): selection strategy `'independent'` or `'subset'` (default: `'independent'`)

    """

    def __init__(self, num_explanations=10, t=0.3, shift_size=0.8, strategy="independent"):

        super().__init__()
        self.num_explanations = num_explanations
        self.t = t
        self.shift_size = shift_size
        self.strategy = strategy

        strategies = ("independent", "subset")
        if strategy not in strategies:
            raise ValueError(f"Strategy must be one of {strategies} got {strategy}")

    @torch.no_grad()
    def explain_batch(self, latents, logits, images, classifier, generator):

        b, c = latents.size()
        labels = logits.argmax(1)

        # initialize set of explanations close to the original examples (same as DICE)
        labels = labels[:, None].repeat(1, self.num_explanations).view(-1)
        z_perturbed = latents[:, None, :].repeat(1, self.num_explanations, 1).view(b, -1, c)
        for i in range(1, self.num_explanations):
            z_perturbed[:, i] = z_perturbed[:, i] + 0.01 * i

        n_images = b * self.num_explanations
        z_perturbed = z_perturbed.view(n_images, c)

        changes = self.__find_style_changes(z_perturbed).cuda()

        diff = self.__compute_diff(z_perturbed, changes, labels, classifier, generator)
        S, D = self.__find_att(diff)

        if self.strategy == "independent":
            top_s = S[:self.num_explanations]
            top_d = D[:self.num_explanations]
            z_perturbed = z_perturbed.view(b, -1, c)
            changes = changes.view(b, -1, c, 2)
            for i, (s, d) in enumerate(zip(top_s, top_d)):
                    z_perturbed[:, i, s] = changes[:, i, s, d]

        if self.strategy == "subset":

            to_explain = list(range(z_perturbed.shape[0]))
            z = z_perturbed.clone()
            for s, d in zip(S, D):

                z = z[to_explain]
                changes = changes[to_explain]
                labels = labels[to_explain]

                z[:, s] = changes[:, s, d]
                z_perturbed[to_explain] = z
                curr_labels = classifier(generator(z)).argmax(1)
                to_explain = torch.where(curr_labels == labels)[0].tolist()

                if not to_explain:
                    break

        return z_perturbed.view(b, -1, c)


    def __compute_diff(self, z_perturbed, changes, labels, classifier, generator):

        n_images, c = z_perturbed.size()
        logits = classifier(generator(z_perturbed))
        targets = 1 - labels
        idxs = torch.arange(n_images)
        z_pos = z_perturbed.clone()
        z_neg = z_perturbed.clone()
        diff = torch.zeros(n_images, c, 2)

        for i in range(c):
            z_neg[:, i] = changes[:, i, 0]
            z_pos[:, i] = changes[:, i, 1]
            diff[:, i, 0] = (classifier(generator(z_neg)) - logits)[idxs, targets]
            diff[:, i, 1] = (classifier(generator(z_pos)) - logits)[idxs, targets]
            z_neg = z_perturbed.clone()
            z_pos = z_perturbed.clone()

        return diff


    def __find_style_changes(self, z_perturbed):

        n_images, c = z_perturbed.size()
        changes = torch.zeros(n_images, c, 2)
        for x, z in enumerate(z_perturbed):
            for s, style in enumerate(z):
                c1 = self.__change_direction(style, s, 0)
                c2 = self.__change_direction(style, s, 1)
                changes[x, s, 0] = c1
                changes[x, s, 1] = c2

        return changes


    def __change_direction(self, coordinate, idx, direction):

        style_min = self.mus_min[idx]
        style_max = self.mus_max[idx]

        target_value = style_min if direction == 0 else style_max
        weight_shift = self.shift_size * (target_value - coordinate)

        coordinate = coordinate * weight_shift

        return coordinate


    def __find_att(self, diff):

        n_images, c, _ = diff.size()
        to_explain = list(range(n_images))
        styles_to_try = list(range(c))
        S = []
        D = []

        while styles_to_try and to_explain:

            mean_diff = diff.mean(0)
            mean_diff = mean_diff[styles_to_try]
            mean_diff[(mean_diff[:, 0] > 0) & (mean_diff[:, 1] > 0)] = 0

            s_index = mean_diff.argmax().item()
            s_max = s_index // 2
            d_max = s_index % 2

            relevant_styles = diff[:, s_max, d_max]
            if (relevant_styles == 0).all():
                break

            S.append(s_max)
            styles_to_try.remove(s_max)
            D.append(d_max)
            to_explain = torch.where(relevant_styles < self.t)[0].tolist()
            diff = diff[to_explain]
            diff[:, s_max] = 0 # remove this style coordinate

        return S, D
