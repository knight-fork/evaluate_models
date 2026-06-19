import os
import torch
import torch.nn as nn

class _LogitsOnly(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out

def get_model(name, num_classes=10, norm='bn'):
    n = name.lower()
    if n in ('wrn_28_10_ln', 'wrn28_10_ln'):
        from .wrn28_10_ln import wrn28_10_ln
        return wrn28_10_ln(num_classes=num_classes)
    if n == 'resnet18':
        from .resnet import ResNet, BasicBlock
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif n == 'resnet34':
        from .resnet import ResNet, BasicBlock
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    elif n == 'mobilenetv2':
        from .mobilenetv2 import MobileNetV2
        model = MobileNetV2(num_classes=num_classes)
    elif n in ('wrn_16_8', 'wrn16_8'):
        from .wide_resnet import Wide_ResNet
        model = Wide_ResNet(16, 8, 0.3, num_classes=num_classes)
    elif n in ('wrn_28_10', 'wrn28_10'):
        from .wide_resnet import Wide_ResNet
        model = Wide_ResNet(28, 10, 0.3, num_classes=num_classes)
    else:
        raise ValueError('unknown cifar10 model: ' + name)
    if norm == 'ln':
        from .norm_utils import replace_bn_with_gn
        model = replace_bn_with_gn(model, num_groups=1)
    return model

def load_checkpoint(model, ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck
    if isinstance(ck, dict):
        for k in ('student_state_dict', 'model_state_dict', 'state_dict', 'net', 'model'):
            if k in ck and isinstance(ck[k], dict):
                sd = ck[k]
                break
    sd = {k[7:] if k.startswith('module.') else k: v for (k, v) in sd.items()}
    target = getattr(model, 'model', model) if isinstance(model, _LogitsOnly) else model
    if sd and all((k.startswith('model.') for k in sd)) and isinstance(model, _LogitsOnly):
        sd = {k[6:]: v for (k, v) in sd.items()}
    target.load_state_dict(sd, strict=True)
    return model
