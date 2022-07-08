from .dive import Dive
from .xgem import Xgem
from .dice import Dice
from .base import ExplainerBase
from .latent_cf import LCF
from .gs import GrowingSpheres
from .stylex import Stylex
from .informed import InformedSearch


__all__ = ["ExplainerBase", "Dice", "Dive", "Stylex", "GrowingSpheres", "LCF", "Xgem"]

def get_explainer(explainer, encoder, generator, val_loader, **kwargs):

    if isinstance(explainer, str):
        explainer = explainer.lower()

        if explainer == "dive":
            return Dive(**kwargs)
        if explainer == "xgem":
            return Xgem(**kwargs)
        if explainer == "is":
            return InformedSearch(encoder, generator, val_loader.dataset, **kwargs)
        if explainer == "dice":
            return Dice(**kwargs)
        if explainer == "gs":
            return GrowingSpheres(**kwargs)
        if explainer == "lcf":
            return LCF(**kwargs)
        if explainer == "stylex":
            return Stylex(**kwargs)

        raise NotImplementedError("Explainer '%s' is not implemented" % explainer)

    return explainer(**kwargs)
