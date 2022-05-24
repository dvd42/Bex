import random
import matplotlib
matplotlib.use("Agg")
from explainability_benchmark import Benchmark
from explainability_benchmark import ExplainerBase
from explainability_benchmark import BasicLogger, WandbLogger
from explainability_benchmark.random_search import EXP_GROUPS


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


corrs = [0.9, 0.95]
clusters = [6, 10]

for corr in corrs:
    for n_clusters in clusters:
        bn = Benchmark(dataset="synbols_font", data_path="explainability_benchmark/data", load_train=False, corr_level=corr, n_clusters_att=n_clusters)
        # bn.runs(EXP_GROUPS["random_search"], log_img_thr=1)
        # bn.run(explainer="ideal")
        bn.run(explainer="dive")
        bn.run(explainer="dice")
        ##XGEM
        bn.run(explainer="dive", lr=0.1, diversity_weight=0, method="none", reconstruction_weight=0.001, lasso_weight=0.1)
        # bn.run(explainer="stylex")
        # bn.run(explainer="lcf")

# bn.run(explainer=MyExplainer, log_images=True, logger=WandbLogger)
# bn.run(explainer="stylex", log_images=True, num_explanations=8, shift_size=1, t=0.5, log_img_thr=1, strategy="subset")
# bn.run(lr=0.1, explainer="dive", method="fisher_spectral_inv", num_explanations=8, diversity_weight=0.1, reconstruction_weight=0.01, lasso_weight=0.0001,
#        log_img_thr=1)
# bn.run(explainer="gs", num_explanations=8, first_radius=5.0, caps=(-1, 1),decrease_radius=5.0, n_candidates=1000, log_img_thr=2.)
print(bn.summarize())
