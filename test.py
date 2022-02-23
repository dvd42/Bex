import random
from explainability_benchmark import Benchmark
from explainability_benchmark import ExplainerBase
from explainability_benchmark import BasicLogger
from explainability_benchmark.random_search import EXP_GROUPS


class MyExplainer(ExplainerBase):

    def __init__(self, data_path, num_explanations=8):

        super().__init__(data_path)
        self.num_explanations = num_explanations
        print("Wow much explainer very counterfactual")


    def explain_batch(self, latents, logits, images, labels, classifier, generator):

        b = latents.shape[0]
        z = latents[:, None, ].repeat(1, self.num_explanations, 1)
        z_perturbed = z + random.random()
        decoded = generator(z_perturbed.view(b * self.num_explanations, -1))
        logits = classifier(decoded)

        return z_perturbed, decoded.view(b, -1, *images.shape[1:])


dummy_explainer = MyExplainer(data_path="explainability_benchmark/data")

bn = Benchmark(data_path="explainability_benchmark/data", load_train=False)
bn.runs(EXP_GROUPS["random_search"], log_img_thr=0.6)
# for exp in EXP_GROUPS["random_search"]:
#     bn.run(log_img_thr=0.5, **exp)


# bn.run(lr=0.01, explainer="dice", yloss_type="hinge_loss", max_iters=500, diversity_weight=10, proximity_weight=1)
# bn.run(lr=0.01, method="fisher_spectral_inv", num_explanations=8, diversity_weight=0.1, reconstruction_weight=5,
#        lasso_weight=5)
# bn.run(lr=0.01, method="dive", num_explanations=8, diversity_weight=0, reconstruction_weight=100,
#        lasso_weight=100)
# bn.run(lr=0.01, method="fisher_spectral_inv", num_explanations=9, diversity_weight=0, reconstruction_weight=0,
        # lasso_regularizer = torch.abs(z_perturbed - latents[:, None, :]).sum()
#        lasso_weight=0)
# bn.run(explainer=dummy_explainer, logger=BasicLogger, output_path="output/test")
# bn.run(explainer=dummy_explainer)
# bn.run(lr=1, method="dive", num_explanations=8)
print(bn.summarize())
