import torch
import torch.nn as nn
import torchvision


def get_model(name, num_classes=100):
    n = name.lower()
    if n in ('resnet18', 'resnet18_timm', 'resnet18_tv'):
        m = torchvision.models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError('unknown imagenet100 model: ' + name)


def load_checkpoint(model, ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck
    if isinstance(ck, dict):
        for k in ('model_state_dict', 'student_state_dict', 'state_dict', 'net', 'model'):
            if k in ck and isinstance(ck[k], dict):
                sd = ck[k]
                break
    new = {}
    for k, v in sd.items():
        for pref in ('module.', '_orig_mod.'):
            if k.startswith(pref):
                k = k[len(pref):]
        new[k] = v
    model.load_state_dict(new, strict=True)
    return model
