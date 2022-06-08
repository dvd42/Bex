from abc import abstractmethod
from tqdm import tqdm
import numpy as np
import h5py
import torch


class ExplainerBase:

    """
    Base class for all explainer methods

    If you wish to test your own explainer on our benchmark use this as a base class and
    override the ``explain_batch`` method

    Example:

        .. code-block:: python

            import random

            class DummyExplainer(ExplainerBase):

                def __init__(self, num_explanations):
                    super().__init__()
                    self.num_explanations = num_explanations

                def explain_batch(self, latents, logits, images, classifier, generator):

                    b = latents.shape[0]
                    # we will produce self.num_explanations counterfactuals per sample
                    z = latents[:, None, :].repeat(1, self.num_explanations, 1)
                    z_perturbed = z + random.random() # create counterfactuals z'

                    return z_perturbed.view(b, self.num_explanations, -1)

            bn = bex.Benchmark()
            bn.run(DummyExplainer, num_explanations=10)

    """

    def __init__(self):

        super().__init__()
        self.data_path = None
        self.num_classes = 2


        self.digest = None


    def _read_cache(self):

        print("Loading from %s" % self.digest)
        self.loaded_data = h5py.File(self.digest, 'a', swmr=True)
        try:
            self.logits = torch.from_numpy(self.loaded_data['val_logits'][...])
            self.mus = torch.from_numpy(self.loaded_data['val_mus'][...])
            self.train_mus = torch.from_numpy(self.loaded_data['train_mus'][...])
            self.latent_mean = self.train_mus.mean(0)
            self.latent_std = self.train_mus.std(0)
            # normalize latents with the training statistics
            self.mus = (self.mus - self.latent_mean) / self.latent_std
            self.mus_min = self.mus.min(0)[0]
            self.mus_max = self.mus.max(0)[0]


        except:
            print("Data not found in the hdf5 cache")


    def _write_cache(self, loader, encoder, generator, classifier, prefix):

        """Loops through the data and stores latents """

        preds = []
        mus = []
        for idx, x, y, categorical_att, continuous_att in tqdm(loader):
            with torch.no_grad():
                x = x.cuda()
                categorical_att = categorical_att.cuda()
                continuous_att = continuous_att.cuda()
                z = encoder(categorical_att, continuous_att)
                mus.append(z.cpu().numpy())

                if "train" not in prefix:
                    reconstruction = generator(z)
                    logits = classifier(reconstruction)
                    preds.append(logits.data.cpu().numpy())

        to_save = dict(mus=np.concatenate(mus, 0))
        if "train" not in prefix:
            to_save["logits"] = np.concatenate(preds, 0)

        with h5py.File(self.digest, 'a') as outfile:
            for k, v in to_save.items():
                outfile[f"{prefix}_{k}"] = v
                print("Done.")


    def _get_latents(self, idx):

        return self.mus[idx].cuda()


    def _get_logits(self, idx):

        return self.logits[idx].cuda()


    def _cleanup(self):

        self.loaded_data.close()


    @abstractmethod
    def explain_batch(self, latents, logits, images, classifier, generator):
        """
        Method to generate a set of counterfactuals for a given batch

        Args:
            latents (``torch.Tensor``): standardized latent :math:`\\textbf{z}` representation of samples to be perturbed
            logits (``torch.Tensor``): classifier logits given :math:`\\textbf{z}`
            images (``torch.Tensor``): images :math:`x` produced by the generator given :math:`\\textbf{z}`
            classifier (``torch.nn.Module``): classifier to explain :math:`\\hat{f}(x)`
            generator (``callable``): function that takes a batch of latents :math:`\\textbf{z'}` and returns a batch of images

        Returns:
            (``torch.Tensor``): the obtained counterfactuals :math:`\\textbf{z'}` for each batch element

        Shape:
            latents :math:`(B, Z)`\n
            logits :math:`(B, 2)`\n
            images :math:`(B, C, H, W)`\n
            obtained counterfactuals: :math:`(B, n\_explanations, Z)`\n

        """
        raise NotImplementedError
