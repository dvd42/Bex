from .dive import Dive



class Xgem(Dive):

    """ xGEM explainer as described in https://arxiv.org/abs/1806.08867

    Args:
        num_explanations (``int``, optional): number of counterfactuals to be generated (default: 10)
        lr (``float``, optional): learning rate (default: 0.1)
        num_iters (``int``, optional): number of gradient descent steps to perform (default: 50)
        reconstruction_weight (``float``, optional): weight of the reconstruction term in the loss function (default: 0.01)
    """

    def __init__(self, num_explanations=10, lr=0.1, num_iters=50, reconstruction_weight=0.001):

        super().__init__()

        self.lr = lr
        self.diversity_weight = 0
        self.lasso_weight = 0
        self.reconstruction_weight = reconstruction_weight
        self.num_iters = num_iters
        self.num_explanations = num_explanations
        self.method ="none"
        self.cache = True


    def explain_batch(self, latents, logits, images, classifier, generator):

        return super().explain_batch(latents, logits, images, classifier, generator)

