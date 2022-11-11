from bex import Benchmark

corrs = [0.5, 0.95]
clusters = [6, 10]

for corr in corrs:
    for n_clusters in clusters:
        bn = Benchmark(corr_level=corr, n_corr=n_clusters)
        bn.runs({"explainer": exp} for exp in ["IS", "dice", "dive", "gs", "stylex", "lcf", "xgem"])
        bn.summarize()
