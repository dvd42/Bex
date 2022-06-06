from .dive import Dive
from .dice import Dice
from .base import ExplainerBase
from .latent_cf import LCF
from .gs import GrowingSpheres
from .stylex import Stylex
from .ideal import IdealExplainer

def get_explainer(explainer, encoder, generator, classifier, train_loader, val_loader, **kwargs):

    if isinstance(explainer, str):
        explainer = explainer.lower()

        if explainer == "dive":
            return Dive(encoder, generator, classifier, train_loader, **kwargs)
        if explainer == "ideal":
            return IdealExplainer(encoder, generator, val_loader.dataset, **kwargs)
        if explainer == "dice":
            return Dice(**kwargs)
        if explainer == "gs":
            return GrowingSpheres(**kwargs)
        if explainer == "lcf":
            return LCF(**kwargs)
        if explainer == "stylex":
            return Stylex(**kwargs)

        raise NotImplementedError("Explainer %s not implemented" % explainer)

    return explainer(**kwargs)
