import torch
import numpy as np

from .base import ExplainerBase, LatentExplainerBase


class GrowingSpheres(ExplainerBase):
    """
    class to fit the Original Growing Spheres algorithm

    Inputs:
    obs_to_interprete: instance whose prediction is to be interpreded
    prediction_fn: prediction function, must return an integer label
    caps: min max values of the explored area. Right now: if not None, the minimum and maximum values of the
    target_class: target class of the CF to be generated. If None, the algorithm will look for any CF that is predicted to belong to a different class than obs_to_interprete
    n_in_layer: number of observations to generate at each step # to do
    layer_shape: shape of the layer to explore the space
    first_radius: radius of the first hyperball generated to explore the space
    decrease_radius: parameter controlling the size of the are covered at each step
    sparse: controls the sparsity of the final solution (boolean)
    verbose: text
    """

    def __init__(self, data_path="data", num_explanations=8, n_candidates=15, layer_shape="sphere", caps=1, first_radius=0.1, decrease_radius=10):

        super().__init__()
        self.num_explanations = num_explanations
        self.n_candidates = int(n_candidates)
        # self.prediction_fn = prediction_fn
        # self.y_obs = prediction_fn(obs_to_interprete.reshape(1, -1))


        self.caps = caps
        self.first_radius = first_radius
        if decrease_radius <= 1.0:
            raise ValueError("Parameter decrease_radius must be > 1.0")
        else:
            self.decrease_radius = decrease_radius

        if layer_shape in ['ring', 'ball', 'sphere']:
            self.layer_shape = layer_shape
        else:
            raise ValueError("Parameter layer_shape must be either 'ring', 'ball' or 'sphere'.")


    @torch.no_grad()
    def explain_batch(self, latents, logits, images, classifier, generator):
        """
        Finds the decision border then perform projections to make the explanation sparse.
        """
        b, c = latents.size()
        counterfactuals = latents.clone()
        counterfactuals = counterfactuals[:, None, :].repeat(1, self.num_explanations, 1)
        for i, (latent, logit) in enumerate(zip(latents, logits)):
            latent = latent[None]
            logit = logit[None]
            enemies_ = self.exploration(latent, logit, generator, classifier)
            if enemies_ is not None:
                closest_enemy_ = sorted(enemies_,
                                        key= lambda x: torch.pairwise_distance(latent.reshape(1, -1), x.reshape(1, -1)))[:self.num_explanations]

                for j, e in enumerate(closest_enemy_):
                    out = self.feature_selection(e, latent, logit, generator, classifier)
                    counterfactuals[i, j] = out

        return counterfactuals


    def exploration(self, latents, logits, generator, classifier):
        """
        Exploration of the feature space to find the decision boundary. Generation of instances in growing hyperspherical layers.
        """
        n_enemies_ = 999
        radius_ = self.first_radius

        while n_enemies_ > 0:

            first_layer_ = self.enemies_in_layer_(latents, logits, generator, classifier, radius=radius_, caps=self.caps, n=self.n_candidates, first_layer=True)

            n_enemies_ = first_layer_.shape[0]
            radius_ = radius_ / self.decrease_radius # radius gets dicreased no matter, even if no enemy?

            if radius_ < 1e-6:
                print("No suitable enemies found")
                return None

        step_ = radius_ * self.decrease_radius

        while n_enemies_ <= 0:

            layer = self.enemies_in_layer_(latents, logits, generator, classifier, layer_shape=self.layer_shape, radius=radius_, step=step_, caps=self.caps,
                                            n=self.n_candidates, first_layer=False)

            n_enemies_ = layer.shape[0]
            radius_ = radius_ + step_

        print("Number of enemies: ", n_enemies_)

        return layer


    def enemies_in_layer_(self, latents, logits, generator, classifier, layer_shape='ring', radius=None, step=None, caps=None, n=1000, first_layer=False):
        """
        Basis for GS: generates a hypersphere layer, labels it with the blackbox and returns the instances that are predicted to belong to the target class.
        """
        # todo: split generate and get_enemies

        if first_layer:
            layer = generate_ball(latents, radius, n)

        else:

            if self.layer_shape == 'ring':
                segment = (radius, radius + step)
                layer = generate_ring(latents, segment, n)

            elif self.layer_shape == 'sphere':
                layer = generate_sphere(latents, radius + step, n)

            elif self.layer_shape == 'ball':
                layer = generate_ball(latents, radius + step, n)

        if caps != 0:
            layer = torch.clamp(layer, caps * -1, caps)

        decoded = generator(layer)
        preds = classifier(decoded).max(1)[1]

        enemies_layer = layer[preds != logits.max(1)[1]]

        return enemies_layer


    def feature_selection(self, counterfactual, latents, logits, generator, classifier):
        """
        Projection step of the GS algorithm. Make projections to make (e* - obs_to_interprete) sparse. Heuristic: sort the coordinates of np.abs(e* - obs_to_interprete) in ascending order and project as long as it does not change the predicted class

        Inputs:
        counterfactual: e*
        """

        move_sorted = sorted(enumerate(abs(counterfactual - latents.flatten())), key=lambda x: x[1])
        move_sorted = [x[0] for x in move_sorted if x[1] > 0.0]

        out = counterfactual.clone()

        reduced = 0

        for k in move_sorted:

            new_enn = out.clone()
            new_enn[k] = latents.flatten()[k]

            decoded = generator(new_enn.view(-1, latents.shape[1]))
            preds = classifier(decoded).max(1)[1]
            condition_class = preds != logits.max(1)[1]

            if condition_class:
                out[k] = new_enn[k]
                reduced += 1


        return out.view(-1, latents.shape[1])


class LatentGrowingSpheres(LatentExplainerBase):
    """
    class to fit the Original Growing Spheres algorithm

    Inputs:
    obs_to_interprete: instance whose prediction is to be interpreded
    prediction_fn: prediction function, must return an integer label
    caps: min max values of the explored area. Right now: if not None, the minimum and maximum values of the
    target_class: target class of the CF to be generated. If None, the algorithm will look for any CF that is predicted to belong to a different class than obs_to_interprete
    n_in_layer: number of observations to generate at each step # to do
    layer_shape: shape of the layer to explore the space
    first_radius: radius of the first hyperball generated to explore the space
    decrease_radius: parameter controlling the size of the are covered at each step
    sparse: controls the sparsity of the final solution (boolean)
    verbose: text
    """

    def __init__(self, num_explanations=8, n_candidates=15, layer_shape="sphere", caps=None, first_radius=0.1, decrease_radius=10):

        super().__init__()
        self.num_explanations = num_explanations
        self.n_candidates = int(n_candidates)
        # self.prediction_fn = prediction_fn
        # self.y_obs = prediction_fn(obs_to_interprete.reshape(1, -1))


        self.caps = caps
        self.first_radius = first_radius
        if decrease_radius <= 1.0:
            raise ValueError("Parameter decrease_radius must be > 1.0")
        else:
            self.decrease_radius = decrease_radius

        if layer_shape in ['ring', 'ball', 'sphere']:
            self.layer_shape = layer_shape
        else:
            raise ValueError("Parameter layer_shape must be either 'ring', 'ball' or 'sphere'.")


    @torch.no_grad()
    def explain_batch(self, latents, logits, classifier):
        """
        Finds the decision border then perform projections to make the explanation sparse.
        """
        b, c = latents.size()
        counterfactuals = latents.clone()
        counterfactuals = counterfactuals[:, None, :].repeat(1, self.num_explanations, 1)
        for i, (latent, logit) in enumerate(zip(latents, logits)):
            latent = latent[None]
            logit = logit[None]
            enemies_ = self.exploration(latent, logit, classifier)
            if enemies_ is not None:
                closest_enemy_ = sorted(enemies_,
                                        key= lambda x: torch.pairwise_distance(latent.reshape(1, -1), x.reshape(1, -1)))[:self.num_explanations]

                for j, e in enumerate(closest_enemy_):
                    out = self.feature_selection(e, latent, logit, classifier)
                    counterfactuals[i, j] = out

        return counterfactuals


    def exploration(self, latents, logits, classifier):
        """
        Exploration of the feature space to find the decision boundary. Generation of instances in growing hyperspherical layers.
        """
        n_enemies_ = 999
        radius_ = self.first_radius

        while n_enemies_ > 0:

            first_layer_ = self.enemies_in_layer_(latents, logits, classifier, radius=radius_, caps=self.caps, n=self.n_candidates, first_layer=True)

            n_enemies_ = first_layer_.shape[0]
            radius_ = radius_ / self.decrease_radius # radius gets dicreased no matter, even if no enemy?

            if radius_ < 1e-6:
                print("No suitable enemies found")
                return None

        step_ = radius_ * self.decrease_radius

        while n_enemies_ <= 0:

            layer = self.enemies_in_layer_(latents, logits, classifier, layer_shape=self.layer_shape, radius=radius_, step=step_, caps=self.caps,
                                            n=self.n_candidates, first_layer=False)

            n_enemies_ = layer.shape[0]
            radius_ = radius_ + step_

        print("Number of enemies: ", n_enemies_)

        return layer


    def enemies_in_layer_(self, latents, logits, classifier, layer_shape='ring', radius=None, step=None, caps=None, n=1000, first_layer=False):
        """
        Basis for GS: generates a hypersphere layer, labels it with the blackbox and returns the instances that are predicted to belong to the target class.
        """
        # todo: split generate and get_enemies

        if first_layer:
            layer = generate_ball(latents, radius, n)

        else:

            if self.layer_shape == 'ring':
                segment = (radius, radius + step)
                layer = generate_ring(latents, segment, n)

            elif self.layer_shape == 'sphere':
                layer = generate_sphere(latents, radius + step, n)

            elif self.layer_shape == 'ball':
                layer = generate_ball(latents, radius + step, n)

        #cap here: not optimal - To do
        if caps != None:
            layer = torch.clamp(caps[0], caps[1])

        preds = classifier(layer).max(1)[1]

        enemies_layer = layer[preds != logits.max(1)[1]]

        return enemies_layer


    def feature_selection(self, counterfactual, latents, logits, classifier):
        """
        Projection step of the GS algorithm. Make projections to make (e* - obs_to_interprete) sparse. Heuristic: sort the coordinates of np.abs(e* - obs_to_interprete) in ascending order and project as long as it does not change the predicted class

        Inputs:
        counterfactual: e*
        """

        move_sorted = sorted(enumerate(abs(counterfactual - latents.flatten())), key=lambda x: x[1])
        move_sorted = [x[0] for x in move_sorted if x[1] > 0.0]

        out = counterfactual.clone()

        reduced = 0

        for k in move_sorted:

            new_enn = out.clone()
            new_enn[k] = latents.flatten()[k]

            preds = classifier(new_enn.view(-1, latents.shape[1])).max(1)[1]
            condition_class = preds != logits.max(1)[1]

            if condition_class:
                out[k] = new_enn[k]
                reduced += 1


        return out.view(-1, latents.shape[1])


def generate_ball(center, r, n):
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

def generate_ring(center, segment, n):
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

def generate_sphere(center, r, n):
    def norm(v):
            return torch.linalg.norm(v, ord=2, dim=1)
    d = center.shape[1]
    # z = np.random.normal(0, 1, (n, d))
    z = torch.zeros(n, d, device=center.device)
    z.normal_(0, 1)
    z = z/(norm(z)[:, None]) * r + center
    return z
