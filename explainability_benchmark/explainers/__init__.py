from .dive import Dive
from .dice import Dice
from .base import ExplainerBase, LatentExplainerBase
from .latent_cf import LCF
from .gs import GrowingSpheres


def get_explainer(explainer, encoder, generator, classifier, train_loader, z_explainer, **kwargs):

    if isinstance(explainer, str):
        explainer = explainer.lower()

        if explainer == "dive":
            return Dive(encoder, generator, classifier, train_loader, **kwargs)
        if explainer == "dice":
            return Dice(**kwargs)
        if explainer == "gs":
            return GrowingSpheres(**kwargs)
        if explainer == "lcf":
            return LCF(**kwargs)

        raise NotImplementedError("Explainer %s not implemented" % explainer)

    return explainer(**kwargs)
