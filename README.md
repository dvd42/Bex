<img src="images/cover.jpg" alt="framework"/>

# A Benchmark For Counterfactual Explainers

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> Explainability methods have been widely used to provide insight into the decisions made by statistical models, thus facilitating their adoption in various domains within the industry. Counterfactual explanation methods aim to improve our understanding of a model by perturbing samples in a way that would alter its response in an unexpected manner. This information is helpful for users and for machine learning practitioners to understand and improve their models. Given the value provided by counterfactual explanations, there is a growing interest in the research community to investigate and propose new methods. However, we identify two issues that could hinder the progress in this field. (1) Existing metrics do not accurately reflect the value of an explainability method for the users. (2) Comparisons between methods are usually performed with datasets like CelebA, where images are annotated with attributes that do not fully describe them and with subjective attributes such as "Attractive". In this work, we address these problems by proposing an evaluation method with a principled metric to evaluate and compare different counterfactual explanation methods. The evaluation is based on a synthetic dataset where images are fully described by their annotated attributes. As a result, we are able to perform a fair comparison of multiple explainability methods in the recent literature, obtaining insights about their performance.

[[paper]](https://openreview.net/pdf?id=RYeRNwRjNE)

## Description

Code repository for the Bex explainability benchmark. Models and datasets that comprise the benchmark can be found [here](https://zenodo.org/record/6616598). They are
automatically downloaded when the benchmark is ran.


The dataset used for the benchmark is a modified version of [Synbols](https://arxiv.org/abs/2009.06415) that contains black and white characters with various attributest that define them (e.g., font, rotaion, scale, etc)

<img src="images/samples.png" alt="samples"/>

## Installation

The recommended way to install Bex is via [PyPI](https://pypi.org/project/bex/)

```bash
pip install bex
```


## Usage

For more information about the usage check out the [Documentation](https://dvd42.github.io/Bex)

We provide a set of counterfactuals explainers already implemented in the benchmark:

1. [Beyond Trivial Counterfactual Explanations with Diverse Valuable Explanations](https://arxiv.org/abs/2103.10226) (DiVE)
2. [xGEMs: Generating Examplars to Explain Black-Box Models](https://arxiv.org/abs/1806.08867) (xGEM)
3. [Latent-CF: A Simple Baseline for Reverse Counterfactual Explanations](https://arxiv.org/abs/2012.09301) (Latent-CF)
4. [Explaining in Style: Training a GAN to explain a classifier in StyleSpace](https://arxiv.org/abs/2104.13369) (StylEx)
5. [Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations](https://arxiv.org/abs/1905.07697) (DiCE)
6. [Inverse Classification for Comparison-based Interpretability in Machine Learning](https://arxiv.org/abs/1712.08443) (Growing Spheres)
7. An oracle with access to the correlated and causal attributes (IS)


The benchmark includes different setting by modifying the number of correlated
`n_corr` attributes and their level of correlation `corr_level`. Right now there are 4 settings available:

* `n_corr=10`, `corr_level=0.95` (default)
* `n_corr=6`, `corr_level=0.95`
* `n_corr=10`, `corr_level=0.5`
* `n_corr=6`, `corr_level=0.5`


### Evaluating one of the predefined explainers

```python
import bex
bn = bex.Benchmark(n_corr=6, corr_level=0.95) # downloads necessary files
bn.run("stylex") # or any of: "dive", "xgem", "lcf", "dice", "gs", "IS" (Oracle)
bn.run("IS", output_path="output/is")
print(bn.summarize()) # get the performance of each explainer as a pandas dataframe
```

You can reproduce the experiments in the paper by runnning `python run_benchmark.py`


### Evaluate a custom explainer

You can evaluate your own explainer like so:

```python
import random
import bex

class DummyExplainer(bex.explainers.ExplainerBase):

    def __init__(self, num_explanations):
        super().__init__()
        self.num_explanations = num_explanations

    # This function describes the behaviour of the custom explainer for a given batch
    def explain_batch(self, latents: torch.Tensor, logits: torch.Tensor,
                      images: torch.Tensor, classifier: torch.nn.Module,
                      generator: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:

        b, c = latents.size()
        # we will produce self.num_explanations counterfactuals per sample
        z = latents[:, None, :].repeat(1, self.num_explanations, 1)
        z_perturbed = z + random.random() # create counterfactuals z'

        return z_perturbed.view(b, self.num_explanations, c)

bn = bex.Benchmark()
bn.run(DummyExplainer, num_explanations=10)
print(bn.summarize()) # get the explainer's performance
```

### Logging

We provide a basic logger to log results and image samples; it is activated by default but you can deactivate it by setting it to `None`:

```python
bn = bex.Benchmark(n_corr=6, corr_level=0.95, logger=None)
bn.run("stylex") # nothing will be logged

# a pandas dataframe holding the results of all .run() calls can always be obtained by calling
bn.summarize()
```

It is also possible to use a custom logger. For example, here is a custom logger using weights and biases:

```python
from bex.loggers import BasicLogger

class WandbLogger(BasicLogger):

    '''
    Args:
        attributes (``Dict``): dictionary containing the run config
        path: (``string``): output path for the logger
        n_batches: (``int``, optional): max number of image batches to log

    '''

    def __init__(self, attributes, path, n_batches=10):

        super().__init__(attributes, path, n_batches)

        wandb.init(project="Synbols-benchmark", dir=self.path, config=self.attributes, reinit=True)


    # accumulate metrics for this step
    def accumulate(self, data, images):

        super().accumulate(data, images)

        wandb.log({f"{k}" :v for k, v in data.items()}, commit=True)


    # log average value of all the metrics across steps
    def log(self):

        self.metrics = {f"{k}_avg": np.mean(v) for k, v in self.metrics.items()}
        wandb.log(self.metrics)

        # create matplotlib figure with the counterfactuals generated
        fig = self.create_cf_figure()

        wandb.log({"Counterfactuals": fig})

        plt.close()

bn = bex.Benchmark(n_corr=6, corr_level=0.95, logger=WandbLogger)
bn.run("IS") # results will be logged to weights and biases

print(bn.summarize()) # results stored in memory
```

## Citation

If you find this work useful, please consider citing the following paper:

Diego Velazquez, Pau Rodriguez, Alexandre Lacoste, Issam H. Laradji, Xavier Roca, and Jordi Gonzàlez. 2023. *Explaining Visual Counterfactual Explainers*. Transactions on Machine Learning Research. ISSN: 2835-8856.


```bibtex
@article{
velazquez2023explaining,
title={Explaining Visual Counterfactual Explainers},
author={Diego Velazquez and Pau Rodriguez and Alexandre Lacoste and Issam H. Laradji and Xavier Roca and Jordi Gonz{\`a}lez},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=RYeRNwRjNE},
note={Reproducibility Certification}
}
```


## Contact

For any bug or feature requests, please create an issue.
