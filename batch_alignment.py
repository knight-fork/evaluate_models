import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchAlignmentWrapper(nn.Module):
    def __init__(self, norm, alpha=1.0, eps=1e-05):
        super().__init__()
        self.norm = norm
        self.alpha = alpha
        self.eps = eps
        self.num_groups = getattr(norm, 'num_groups', 1)
        self.has_affine = getattr(norm, 'weight', None) is not None

    def forward(self, x):
        x_ln = self.norm(x)
        if x.size(0) <= 1 or self.alpha <= 0.0:
            return x_ln
        x_normed = F.group_norm(x, num_groups=self.num_groups, weight=None, bias=None, eps=self.eps)
        batch_mean = x_normed.mean(dim=[0, 2, 3], keepdim=True)
        batch_var = x_normed.var(dim=[0, 2, 3], keepdim=True, unbiased=False)
        x_aligned = (x_normed - batch_mean) / torch.sqrt(batch_var + self.eps)
        if self.has_affine:
            x_aligned = x_aligned * self.norm.weight.view(1, -1, 1, 1) + self.norm.bias.view(1, -1, 1, 1)
        if self.alpha >= 1.0:
            return x_aligned
        return (1.0 - self.alpha) * x_ln + self.alpha * x_aligned


def add_batch_alignment(module, alpha=1.0, norm_types=None):
    if norm_types is None:
        norm_types = (nn.GroupNorm, nn.LayerNorm)
    for name, child in module.named_children():
        if isinstance(child, norm_types) or type(child).__name__ == 'LayerNorm2d':
            setattr(module, name, BatchAlignmentWrapper(child, alpha=alpha))
        else:
            add_batch_alignment(child, alpha=alpha, norm_types=norm_types)
    return module


def count_batch_aligned(module):
    return sum(1 for m in module.modules() if isinstance(m, BatchAlignmentWrapper))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from models.cifar100 import get_model

    teacher = get_model('cifar_wrn_28_10_ln', num_classes=100)
    teacher = add_batch_alignment(teacher, alpha=0.5)
    teacher.train()
    print('batch-aligned layers:', count_batch_aligned(teacher))
    x = torch.randn(8, 3, 32, 32)
    print('output shape:', tuple(teacher(x).shape))
