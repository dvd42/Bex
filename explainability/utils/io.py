import torch


def join_strings(base_string, strings):
  return base_string.join([item for item in strings if item])


def load_weights(G, D, state_dict, weights_root, experiment_name, name_suffix=None, G_ema=None, strict=True, load_optim=True):

    root = '/'.join([weights_root, experiment_name])
    if name_suffix:
        print('Loading %s weights from %s...' % (name_suffix, root))
    else:
        print('Loading weights from %s...' % root)
    if G is not None:
        G.load_state_dict(
        torch.load('%s/%s.pth' % (root, join_strings('_', ['G', name_suffix]))),
        strict=strict)
        if load_optim:
            G.optim.load_state_dict(torch.load('%s/%s.pth' % (root, join_strings('_', ['G_optim', name_suffix]))))
    if D is not None:
        D.load_state_dict(
        torch.load('%s/%s.pth' % (root, join_strings('_', ['D', name_suffix]))),
        strict=strict)
        if load_optim:
            D.optim.load_state_dict(torch.load('%s/%s.pth' % (root, join_strings('_', ['D_optim', name_suffix]))))
    # Load state dict
    for item in state_dict:
        state_dict[item] = torch.load('%s/%s.pth' % (root, join_strings('_', ['state_dict', name_suffix])))[item]
    if G_ema is not None:
        G_ema.load_state_dict(
        torch.load('%s/%s.pth' % (root, join_strings('_', ['G_ema', name_suffix]))),
        strict=strict)

