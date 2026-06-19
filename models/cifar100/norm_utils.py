import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-05, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        u = x.mean([1, 2, 3], keepdim=True)
        v = x.var([1, 2, 3], keepdim=True, unbiased=False)
        x = (x - u) / torch.sqrt(v + self.eps)
        if self.affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


def replace_bn_with_gn(module: nn.Module, num_groups: int, name: str='') -> nn.Module:
    module_output = module
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        num_features = module.num_features
        g = num_groups
        if g > 1 and num_features % g != 0:
            for candidate in range(min(g, num_features), 0, -1):
                if num_features % candidate == 0:
                    g = candidate
                    break
        if g == 1:
            module_output = LayerNorm2d(num_features, eps=module.eps, affine=module.affine)
        else:
            module_output = nn.GroupNorm(g, num_features, eps=module.eps, affine=module.affine)
    for (child_name, child) in module.named_children():
        full_name = f'{name}.{child_name}' if name else child_name
        module_output.add_module(child_name, replace_bn_with_gn(child, num_groups=num_groups, name=full_name))
    return module_output


def replace_bn_with_ln(module: nn.Module, name: str='') -> nn.Module:
    return replace_bn_with_gn(module, num_groups=1, name=name)
