from copy import deepcopy
import numpy as np

dive = {
        "explainer": "dive",
        "lr": 0.01,
        "max_iters": 20,
        "num_explanations": 8,
        "method": "dive",
        "reconstruction_weight": 1,
        "lasso_weight": 1.,
        "diversity_weight": 1,
       }

dice = {
        "explainer": "dice",
        "lr": 0.01,
        "max_iters": 500,
        "proximity_weight": 0.5,
        "num_explanations": 8,
        "diversity_weight": 1,
       }

gs = { "explainer": "gs",
       "first_radius": 0.1,
       "num_explanations": 8,
       "decrease_radius": 10,
      }

lcf = { "explainer": "lcf",
        "lr": 0.1,
        "num_explanations": 8,
        "tolerance": 0.3,
        "max_iters": 500,
        "p": 0.5
      }

np.random.seed(0)

random_search = dive # default hparams
n_trials = 5
lasso_space = [1, 5, 10]
lr_space = [0.01, 0.05, 0.1]
diversity_space = [1, 5, 10]
reconstruction_space = [1, 5, 10]
methods = ['fisher_spectral', 'random', 'dive', 'fisher_spectral_inv']

exps = []
base_exp = deepcopy(random_search)
for run in range(n_trials):
    base_exp['lr'] = float(np.random.choice(lr_space))
    base_exp['diversity_weight'] = float(np.random.choice(diversity_space))
    base_exp['lasso_weight'] = float(np.random.choice(lasso_space))
    base_exp['reconstruction_weight'] = float(np.random.choice(reconstruction_space))
    methods = ["fisher_spectral_inv", "fisher_spectral", "dive", "random"]
    for method in methods:
        method_exp = deepcopy(base_exp)
        method_exp['method'] = method
        if method_exp in exps:
            continue
        exps.append(method_exp)


random_search = dice # default hparams
n_trials = 20
lr_space = [0.01, 0.05, 0.1]
diversity_space = [0.1, 1, 5, 10]
max_iters_space = [500, 1000]
proximity_space = [0.5, 0.1, 1, 10]

base_exp = deepcopy(random_search)
for run in range(n_trials):
    base_exp['lr'] = float(np.random.choice(lr_space))
    base_exp['diversity_weight'] = float(np.random.choice(diversity_space))
    base_exp['proximity_weight'] = float(np.random.choice(proximity_space))
    base_exp['max_iters'] = np.random.choice(max_iters_space)
    exps.append(base_exp)

random_search = gs
n_trials = 20
candidate_space = [5, 15, 100, 1000]
first_radius_space = [0.1, 0.5, 1, 5, 10]
decrease_radius_space = [2, 5, 10, 15]
caps_space =[None, (-1, 1), (-5, 5)]

base_exp = deepcopy(random_search)
for run in range(n_trials):
    base_exp['n_candidates'] = np.random.choice(candidate_space)
    base_exp['first_radius'] = float(np.random.choice(first_radius_space))
    base_exp['decrease_radius'] = float(np.random.choice(decrease_radius_space))
    base_exp['caps'] = np.random.choice(np.array(caps_space, dtype=object))
    exps.append(base_exp)

random_search = lcf
n_trials = 20
lr_space = [0.05, 0.1, 0.2]
p_space = [0.3, 0.5, 0.7]
tolerance_space = [0.1, 0.3, 0.5]
max_iters_space = [500, 1000]

base_exp = deepcopy(random_search)
for run in range(n_trials):
    base_exp['lr'] = float(np.random.choice(lr_space))
    base_exp['max_iters'] = np.random.choice(max_iters_space)
    base_exp['p'] = float(np.random.choice(p_space))
    base_exp["tolerance"] = float(np.random.choice(tolerance_space))
    exps.append(base_exp)

EXP_GROUPS = {}
EXP_GROUPS['random_search'] = exps
