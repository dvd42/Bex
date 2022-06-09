import torch
import numpy as np

from .base import ExplainerBase


class GrowingSpheres(ExplainerBase):

    """ Growing Spheres explainer as described in https://arxiv.org/abs/1712.08443

    num_explanations (``int``, optional): number of counterfactuals to be generated (default: 10)

    n_candidates (``int``, optional): number of observations :math:`n` to generate at each step (default: 50)
    first_radius(``float``, optional): radius :math:`\\eta` of the first hyperball generated (default: 10.0)
    decrease_radius(``float``, optional): parameter controlling the size of the radius at each step (default: 2.0)
    """

    def __init__(self, num_explanations=10, n_candidates=50, first_radius=10, decrease_radius=2):

        super().__init__()
        self.num_explanations = num_explanations
        self.n_candidates = int(n_candidates)

        self.caps = None
        self.first_radius = first_radius
        if decrease_radius <= 1.0:
            raise ValueError("Parameter decrease_radius must be > 1.0")
        else:
            self.decrease_radius = decrease_radius

        self.layer_shape = "sphere"


    @torch.no_grad()
    def explain_batch(self, latents, logits, images, classifier, generator):

        b, c = latents.size()
        counterfactuals = latents.clone()
        counterfactuals = counterfactuals[:, None, :].repeat(1, self.num_explanations, 1)
        for i, (latent, logit) in enumerate(zip(latents, logits)):
            latent = latent[None]
            logit = logit[None]
            enemies_ = self.__exploration(latent, logit, generator, classifier)
            if enemies_ is not None:
                closest_enemy_ = sorted(enemies_,
                                        key= lambda x: torch.pairwise_distance(latent.reshape(1, -1), x.reshape(1, -1)))[:self.num_explanations]

                closest_enemy_ = torch.stack(closest_enemy_)
                # torch.manual_seed(0)
                # closest_enemy_ = torch.randn_like(closest_enemy_)
                # for j, e in enumerate(closest_enemy_):
                #     out = self.feature_selection(e, latent, logit, generator, classifier)
                #     counterfactuals[i, j] = out
                counterfactuals[i, :closest_enemy_.shape[0]] = closest_enemy_
                out = self.__feature_selection(counterfactuals[i], latent, logit, generator, classifier)
                counterfactuals[i, :] = out

        return counterfactuals


    def __exploration(self, latents, logits, generator, classifier):
        """
        Exploration of the feature space to find the decision boundary. Generation of instances in growing hyperspherical layers.
        """
        n_enemies_ = 999
        radius_ = self.first_radius

        while n_enemies_ > 0:

            first_layer_ = self.__enemies_in_layer_(latents, logits, generator, classifier, radius=radius_, caps=self.caps, n=self.n_candidates, first_layer=True)

            n_enemies_ = first_layer_.shape[0]
            radius_ = radius_ / self.decrease_radius # radius gets dicreased no matter, even if no enemy?

            if radius_ < 1e-6:
                return None

        step_ = radius_ * self.decrease_radius

        while n_enemies_ <= 0:

            layer = self.__enemies_in_layer_(latents, logits, generator, classifier, layer_shape=self.layer_shape, radius=radius_, step=step_, caps=self.caps,
                                            n=self.n_candidates, first_layer=False)

            n_enemies_ = layer.shape[0]
            radius_ = radius_ + step_

        return layer


    def __enemies_in_layer_(self, latents, logits, generator, classifier, layer_shape='ring', radius=None, step=None, caps=None, n=1000, first_layer=False):
        """
        Basis for GS: generates a hypersphere layer, labels it with the blackbox and returns the instances that are predicted to belong to the target class.
        """
        # todo: split generate and get_enemies

        if first_layer:
            layer = self.__generate_ball(latents, radius, n)

        else:

            if self.layer_shape == 'ring':
                segment = (radius, radius + step)
                layer = self.__generate_ring(latents, segment, n)

            elif self.layer_shape == 'sphere':
                layer = self.__generate_sphere(latents, radius + step, n)

            elif self.layer_shape == 'ball':
                layer = self.__generate_ball(latents, radius + step, n)

        if caps is not None:
            layer = torch.clamp(layer, caps * -1, caps)

        decoded = generator(layer)
        preds = classifier(decoded).max(1)[1]

        enemies_layer = layer[preds != logits.max(1)[1]]

        return enemies_layer


    def __feature_selection(self, counterfactual, latents, logits, generator, classifier):
        """
        Projection step of the GS algorithm. Make projections to make (e* - obs_to_interprete) sparse. Heuristic: sort the coordinates of np.abs(e* - obs_to_interprete) in ascending order and project as long as it does not change the predicted class

        Inputs:
        counterfactual: e*
        """

        ne, c = counterfactual.size()
        counterfactual = counterfactual.view(-1, c)
        latents = latents.repeat(ne, 1).view(-1, c)
        logits = logits.repeat(ne, 1).view(-1, logits.shape[1])
        perturbations = (counterfactual - latents).abs()
        move_sorted = torch.argsort(perturbations).t()

        out = counterfactual.clone()

        arange = list(range(ne))

        for k in move_sorted:
            z = out.clone()
            z[arange, k] = latents[arange, k]
            curr_labels = classifier(generator(z)).argmax(1)
            change = curr_labels != logits.max(1)[1]
            out[change, k[change]] = z[change, k[change]]

        # out2 = counterfactual[0].clone()
        # latents = latents[0][None]
        # move_sorted = sorted(enumerate(abs(out2 - latents.flatten())), key=lambda x: x[1])
        # move_sorted = [x[0] for x in move_sorted if x[1] > 0.0]
        # logits = logits[0][None]
        # for k in move_sorted:

        #     new_enn = out2.clone()
        #     new_enn[k] = latents.flatten()[k]

        #     decoded = generator(new_enn.view(-1, latents.shape[1]))
        #     preds = classifier(decoded).max(1)[1]
        #     condition_class = preds != logits.max(1)[1]

        #     if condition_class:
        #         out2[k] = new_enn[k]


        return out.view(-1, latents.shape[1])


    def __generate_ball(self, center, r, n):
        def norm(v):
            return torch.linalg.norm(v, ord=2, dim=1)
        d = center.shape[1]
        # u = np.random.normal(0,1,(n, d+2))  # an array of (d+2) normally distributed random variables
        u = torch.zeros(n, d+2, device=center.device)
        u.normal_(0, 1)
        norm_ = norm(u)
        u = 1/norm_[:,None]* u
        x = u[:, 0:d] * r #take the first d coordinates
        x = x + center
        return x


    def __generate_ring(self, center, segment, n):
        def norm(v):
            return torch.linalg.norm(v, ord=2, dim=1)
        d = center.shape[1]
        # z = np.random.normal(0, 1, (n, d))
        z = torch.zeros(n, d, device=center.device)
        z.normal_(0, 1)
        try:
            u = torch.zeros(n, device=center.device)
            u.uniform_(segment[0]**d, segment[1]**d)
        except OverflowError:
            raise OverflowError("Dimension too big for hyperball sampling. Please use layer_shape='sphere' instead.")
        r = u**(1/float(d))
        # z = torch.tensor([a * b / c for a, b, c in zip(z, r,  norm(z))])
        z = z * r[:, None] / norm(z)[:, None]
        z = z + center
        return z


    def __generate_sphere(self, center, r, n):
        def norm(v):
                return torch.linalg.norm(v, ord=2, dim=1)
        d = center.shape[1]
        # z = np.random.normal(0, 1, (n, d))
        z = torch.zeros(n, d, device=center.device)
        z.normal_(0, 1)
        z = z/(norm(z)[:, None]) * r + center
        return z
