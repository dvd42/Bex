from .dive import Dive
from .dice import Dice
from .base import ExplainerBase


def get_explainer(explainer, encoder, generator, classifier, train_loader, **kwargs):

    if isinstance(explainer, str):
        explainer = explainer.lower()

        if explainer == "dive":
            return Dive(encoder, generator, classifier, train_loader, **kwargs)
        if explainer == "dice":
            return Dice(encoder, generator, classifier, train_loader, **kwargs)

        raise ValueError("Explainer %s not found" % explainer)

    else:
        return explainer
