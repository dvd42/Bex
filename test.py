import random
import wandb
from Bex import Benchmark
from Bex.explainers import ExplainerBase
from Bex.loggers import BasicLogger


class WandbLogger(BasicLogger):


    def __init__(self, attributes, path, n_batches=10):

        super().__init__(attributes, path, n_batches)

        wandb.init(project="Synbols-benchmark", dir=self.path, config=self.attributes, reinit=True,
                   tags=["test"])


    def accumulate(self, data, images):

        super().accumulate(data, images)

        wandb.log({f"{k}" :v for k, v in data.items()}, commit=True)


    def log(self):

        self.metrics = {f"{k}_avg": np.mean(v) for k, v in self.metrics.items()}
        wandb.log(self.metrics)

        self.prepare_images_to_log()

        wandb.log({"Counterfactuals": self._figure})


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



# corrs = [0.9, 0.95]
corrs = [0.95]
clusters = [6]

import numpy as np
import torch
for _ in range(1):
    for corr in corrs:
        for n_clusters in clusters:
            bn = Benchmark(corr_level=corr, n_corr=n_clusters)
            # bn.run("stylex")
            # bn.run("xgem")
            bn.run("ideal", logger=WandbLogger)
            # bn.run("dive")
            # bn.run("dice")
            # bn.run("lcf")
            # bn.run("gs")
            print(bn.summarize())

# bn = Benchmark(dataset="synbols_font", data_path="explainability_benchmark/data", corr_level=0.95, n_clusters_att=10)
# bn.run(explainer="dive", lr=0.1, diversity_weight=0, method="none", reconstruction_weight=0.01, lasso_weight=0)
# bn.run("stylex")
# bn.run("dice")
# bn.run("lcf")
# bn.runs(EXP_GROUPS["random_search"], log_img_thr=1)
# bn.run(explainer="ideal")
#bn.run(explainer="stylex", output_path="images/stylex", logger=BasicLogger)
# bn.run(explainer="dive", lr=0.1, method="fisher_spectral_inv", lasso_weight=0.001, reconstruction_weight=0.0001, diversity_weight=0)
# bn.run(explainer="dive", lr=1)
        # bn.run(explainer="gs")
        # bn.run(explainer="dive")
# bn.run(explainer="dice", lr=0.1, diversity_weight=1, proximity_weight=1)
#         # bn.run(explainer="stylex")
        # bn.run(explainer="lcf")
