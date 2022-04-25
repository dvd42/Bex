import random
import matplotlib
matplotlib.use("Agg")
from explainability_benchmark import Benchmark
from explainability_benchmark import ExplainerBase, LatentExplainerBase
from explainability_benchmark import BasicLogger, WandbLogger
from explainability_benchmark.random_search import EXP_GROUPS

class LatentMyExplainer(LatentExplainerBase):

    def __init__(self, num_explanations=8):

        super().__init__()
        self.num_explanations = num_explanations
        print("Wow much explainer very counterfactual")


    def explain_batch(self, latents, logits, classifier):

        b = latents.shape[0]
        z = latents[:, None, ].repeat(1, self.num_explanations, 1)
        z_perturbed = z + random.random()

        return z_perturbed


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

# bn = Benchmark(dataset="uniform_z", classifier="mlp", data_path="explainability_benchmark/data", load_train=True)

bn = Benchmark(dataset="synbols", data_path="explainability_benchmark/data", load_train=False)
bn.runs(EXP_GROUPS["random_search"], log_img_thr=2.)

# bn.run(explainer=MyExplainer, log_images=True, logger=WandbLogger)
# bn.run(explainer="lcf")
# bn.run(lr=0.05, explainer="dive", method="fisher_spectral", num_explanations=8, diversity_weight=5, reconstruction_weight=5,
#        lasso_weight=5)
# bn.run(explainer=LatentMyExplainer, z_explainer=True, logger=BasicLogger)
# bn.run(explainer="gs", num_explanations=8, first_radius=5.0, decrease_radius=5.0, n_candidates=1000, log_img_thr=2.)
# bn.run(explainer="dice", num_explanations=8, first_radius=5.0, decrease_radius=5.0, z_explainer=True, n_candidates=1000)
# bn.run(lr=0.01, method="fisher_spectral_inv", num_explanations=8, diversity_weight=100, reconstruction_weight=5,
       # lasso_weight=5)
# bn.run(lr=0.01, method="fisher_spectral_inv", num_explanations=9, diversity_weight=0, reconstruction_weight=0,
        # lasso_regularizer = torch.abs(z_perturbed - latents[:, None, :]).sum()
#        lasso_weight=0)
# bn.run(explainer=dummy_explainer, logger=BasicLogger, output_path="output/test")
# bn.run(explainer=dummy_explainer)
# bn.run(lr=1, method="dive", num_explanations=8)
print(bn.summarize())
