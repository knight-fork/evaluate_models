import os
import torch
import torch.nn as nn


def get_model(name, num_classes=10):
    n = name.lower()
    if n == 'resnet18':
        from .resnet import ResNet, BasicBlock
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if n == 'mobilenetv2':
        from .mobilenetv2 import MobileNetV2
        return MobileNetV2(num_classes=num_classes)
    if n in ('wrn_16_8', 'wrn16_8', 'wideresnet16_8'):
        from .wide_resnet import Wide_ResNet
        return Wide_ResNet(16, 8, 0.3, num_classes=num_classes)
    raise ValueError('unknown cifar10 model: ' + name)


def load_checkpoint(model, ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck
    if isinstance(ck, dict):
        for k in ('student_state_dict', 'model_state_dict', 'state_dict', 'net', 'model'):
            if k in ck and isinstance(ck[k], dict):
                sd = ck[k]
                break
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model
