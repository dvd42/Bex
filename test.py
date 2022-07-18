import random
from WhyY import Benchmark
from WhyY import ExplainerBase
from WhyY import BasicLogger, WandbLogger
from WhyY.random_search import EXP_GROUPS


class MyExplainer(ExplainerBase):

    def __init__(self, num_explanations=8):

        super().__init__()
        self.num_explanations = num_explanations
        print("Wow much explainer very counterfactual")


    def explain_batch(self, latents, logits, images, classifier, generator):

        b = latents.shape[0]
        z = latents[:, None, ].repeat(1, self.num_explanations, 1)
        z_perturbed = z + random.random()
        decoded = generator(z_perturbed.view(b * self.num_explanations, -1))
        logits = classifier(decoded)

        return z_perturbed


dummy_explainer = MyExplainer()


# corrs = [0.9, 0.95]
corrs = [0.95, 0.5]
clusters = [10, 6]

import numpy as np
import torch
# for _ in range(1):
#     for corr in corrs:
#         for n_clusters in clusters:
#             bn = Benchmark(dataset="synbols_font", data_path="data", corr_level=corr, n_clusters_att=n_clusters)
            # bn.run("ideal")
            # bn.run("stylex")
            # bn.run("xgem")
            # bn.run("dive", sparsity_weight=0.)
            # bn.run("dice")
            #XGEM
            # bn.run(explainer="dive", lr=0.1, diversity_weight=0, method="none", reconstruction_weight=0.01, lasso_weight=0)
            # bn.run("lcf")
            # bn.run("gs")
bn = Benchmark(dataset="synbols_font", data_path="data", corr_level=0.95, n_clusters_att=10)
bn.runs(EXP_GROUPS["random_search"], log_img_thr=1.)
# bn.run(explainer="dive", sparsity_weight=0, beta=0.2, lasso_weight=10, reconstruction_weight=10, diversity_weight=10)
# bn.run("stylex")
# bn.run("dice")
# bn.run("lcf")
# bn.runs(EXP_GROUPS["random_search"], log_img_thr=1)
# bn.run(explainer="ideal")
#bn.run(explainer="stylex", output_path="images/stylex", logger=BasicLogger)
# bn.run(explainer="dive", lr=0.1, method="fisher_spectral_inv", lasso_weight=0.001, reconstruction_weight=0.0001, diversity_weight=0)
# bn.run(explainer="dive", diversity_weight=0.001, lasso_weight=0.1, reconstruction_weight=0.0001)
        # bn.run(explainer="gs")
        # bn.run(explainer="dive")
# bn.run(explainer="dice", lr=0.1, diversity_weight=1, proximity_weight=1)
#         # bn.run(explainer="stylex")
        # bn.run(explainer="lcf")

print(bn.summarize())
