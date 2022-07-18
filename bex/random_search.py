from copy import deepcopy
import numpy as np

dive = {
        "explainer": "dive",
        "lr": 0.01,
        "max_iters": 50,
        "method": "dive",
        "reconstruction_weight": 1,
        "lasso_weight": 1.,
        "diversity_weight": 1,
       }

xgem = {
        "explainer": "xgem",
        "lr": 0.01,
        "max_iters": 50,
        "reconstruction_weight": 1,
       }

dice = {
        "explainer": "dice",
        "lr": 0.01,
        "max_iters": 50,
        "proximity_weight": 0.5,
        "diversity_weight": 1,
       }

gs = { "explainer": "gs",
       "first_radius": 0.1,
      "caps": None,
       "decrease_radius": 10,
      }

lcf = { "explainer": "lcf",
        "lr": 0.1,
        "tolerance": 0.3,
        "max_iters": 50,
        "p": 0.5
      }


stylex = { "explainer": "stylex",
          "t": 0.1,
          "shift_size": 2,
          "strategy": "independent"
          }


# np.random.seed(0)

random_search = dive # default hparams
n_trials = 40
lasso_space = list([0.0001, 0.001, 0.01, 0.1, 1, 10])
lr_space = [0.001, 0.01, 0.05, 0.1]
diversity_space = [0] + list([0.001, 0.01, 0.1, 1, 10])
reconstruction_space = [0] + list([0.0001, 0.001, 0.01, 0.1, 1, 10])

exps = []
base_exp = deepcopy(random_search)
for run in range(n_trials):
    base_exp['lr'] = float(np.random.choice(lr_space))
    base_exp['diversity_weight'] = float(np.random.choice(diversity_space))
    base_exp['lasso_weight'] = float(np.random.choice(lasso_space))
    base_exp['reconstruction_weight'] = float(np.random.choice(reconstruction_space))
    methods = ["fisher_spectral"]
    for method in methods:
        method_exp = deepcopy(base_exp)
        method_exp['method'] = method
        if method_exp in exps:
            continue
        exps.append(deepcopy(method_exp))


# random_search = dice # default hparams
# n_trials = 40
# proximity_space = reconstruction_space

# base_exp = deepcopy(random_search)
# for run in range(n_trials):
#     base_exp['lr'] = float(np.random.choice(lr_space))
#     base_exp['diversity_weight'] = float(np.random.choice(diversity_space))
#     base_exp['proximity_weight'] = float(np.random.choice(proximity_space))
#     exps.append(deepcopy(base_exp))


# random_search = stylex
# n_trials = 40
# t_space = [0.1, 0.3, 0.5, 0.7]
# shift_size_space = [0.1, 0.5, 1, 1.5, 2]
# base_exp = deepcopy(random_search)
# for run in range(n_trials):
#     base_exp['t'] = float(np.random.choice(t_space))
#     base_exp['shift_size'] = float(np.random.choice(shift_size_space))

#     strategies = ["subset", "independent"]
#     for strategy in strategies:
#         strategy_exp = deepcopy(base_exp)
#         strategy_exp['strategy'] = strategy
#         if strategy_exp in exps:
#             continue
#         exps.append(deepcopy(strategy_exp))


# random_search = gs
# n_trials = 40
# candidate_space = [15, 50, 100, 1000]
# first_radius_space = [0.1, 0.5, 1, 5, 10]
# decrease_radius_space = [2, 5, 10, 15]
# caps_space =[None, 1, 5, 10]

# base_exp = deepcopy(random_search)
# for run in range(n_trials):
#     base_exp['n_candidates'] = np.random.choice(candidate_space)
#     base_exp['first_radius'] = float(np.random.choice(first_radius_space))
#     base_exp['decrease_radius'] = float(np.random.choice(decrease_radius_space))
#     base_exp['caps'] = np.random.choice(np.array(caps_space, dtype=object))
#     exps.append(deepcopy(base_exp))

# random_search = lcf
# n_trials = 40
# p_space = [0, 0.1, 0.5, 0.7]
# tolerance_space = [0, 0.1, 0.3, 0.5, 0.7]

# base_exp = deepcopy(random_search)
# for run in range(n_trials):
#     base_exp['lr'] = float(np.random.choice(lr_space))
#     base_exp['p'] = float(np.random.choice(p_space))
#     base_exp["tolerance"] = float(np.random.choice(tolerance_space))
#     exps.append(deepcopy(base_exp))

random_search = xgem
n_trials = 40

exps = []
base_exp = deepcopy(random_search)
for run in range(n_trials):
    base_exp['lr'] = float(np.random.choice(lr_space))
    base_exp['reconstruction_weight'] = float(np.random.choice(reconstruction_space))
    exps.append(deepcopy(base_exp))

print(exps)
EXP_GROUPS = {}
EXP_GROUPS['random_search'] = exps
