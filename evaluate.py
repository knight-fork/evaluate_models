import argparse
import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CIFAR10_MODELS = ('resnet18', 'mobilenetv2', 'wrn_16_8')
CIFAR100_MODELS = ('resnet18', 'wrn_16_8')
IMAGENET100_MODELS = ('resnet18',)


def build_loader(data, data_path, image_size, batch_size, num_workers):
    if data == 'cifar10':
        tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
        ds = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=tf)
        return DataLoader(ds, batch_size=256, shuffle=False, num_workers=num_workers), 10
    if data == 'cifar100':
        tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
        ds = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=tf)
        return DataLoader(ds, batch_size=256, shuffle=False, num_workers=num_workers), 100
    if data == 'imagenet100':
        resize = int(round(image_size * 256 / 224))
        tf = T.Compose([T.Resize(resize), T.CenterCrop(image_size), T.ToTensor(),
                        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        root = os.path.join(data_path, 'val') if os.path.isdir(os.path.join(data_path, 'val')) else data_path
        ds = torchvision.datasets.ImageFolder(root, transform=tf)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers), 100
    raise ValueError('unknown data: ' + data)


def build_model(data, model, num_classes):
    if data == 'cifar10':
        from models.cifar10 import get_model, load_checkpoint
    elif data == 'cifar100':
        from models.cifar100 import get_model, load_checkpoint
    elif data == 'imagenet100':
        from models.imagenet100 import get_model, load_checkpoint
    else:
        raise ValueError('unknown data: ' + data)
    return get_model(model, num_classes=num_classes), load_checkpoint


@torch.no_grad()
def evaluate(model, loader, device, train_mode):
    model.train(train_mode)
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total, correct, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, choices=['cifar10', 'cifar100', 'imagenet100'])
    ap.add_argument('--model', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data_path', default=None)
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    allowed = {'cifar10': CIFAR10_MODELS, 'cifar100': CIFAR100_MODELS, 'imagenet100': IMAGENET100_MODELS}
    if args.model.lower() not in allowed[args.data]:
        raise ValueError('model ' + args.model + ' not supported for ' + args.data +
                         '; choose from ' + str(allowed[args.data]))

    if args.data_path is None:
        if args.data == 'imagenet100':
            raise ValueError('--data_path is required for imagenet100')
        args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    device = args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    num_classes = 10 if args.data == 'cifar10' else 100

    model, load_checkpoint = build_model(args.data, args.model, num_classes)
    model = load_checkpoint(model, args.checkpoint)
    model = model.to(device)

    loader, _ = build_loader(args.data, args.data_path, args.image_size,
                             args.batch_size, args.num_workers)

    train_mode = args.data in ('cifar10', 'cifar100')
    acc, correct, total = evaluate(model, loader, device, train_mode)
    print('{:.4f}'.format(acc))


if __name__ == '__main__':
    main()
