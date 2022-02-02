from .dive import GradientAttack


def get_explainer(exp_dict, **kwargs):
    if exp_dict["explainer"] == "dive":
        return GradientAttack(exp_dict, **kwargs)
    else:
        raise ValueError("Explainer %s not found" %exp_dict["explainer"])
