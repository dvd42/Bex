from . import exp_configs_oracle, exp_configs_explainers, exp_configs_generator, exp_configs_classifier

EXP_GROUPS = {}
EXP_GROUPS.update(exp_configs_oracle.EXP_GROUPS)
EXP_GROUPS.update(exp_configs_explainers.EXP_GROUPS)
EXP_GROUPS.update(exp_configs_generator.EXP_GROUPS)
EXP_GROUPS.update(exp_configs_classifier.EXP_GROUPS)
