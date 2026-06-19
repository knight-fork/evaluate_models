#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

CKPT=./checkpoints
IMAGENET_VAL=./val5000

echo "cifar10  resnet18    (r34->r18)        $(python evaluate.py --data cifar10 --model resnet18    --checkpoint $CKPT/cifar10_res34_res18_bn_exp.pth)"
echo "cifar10  mobilenetv2 (r34->mv2)        $(python evaluate.py --data cifar10 --model mobilenetv2 --checkpoint $CKPT/cifar10_res34_mv2_bn_exp.pth)"
echo "cifar10  wrn_16_8    (wrn28->wrn16_8)  $(python evaluate.py --data cifar10 --model wrn_16_8    --checkpoint $CKPT/cifar10_wrn28_10_wrn_16_8_bn_exp.pth)"
echo "cifar10  resnet18    (wrn28->r18)      $(python evaluate.py --data cifar10 --model resnet18    --checkpoint $CKPT/cifar10_wrn_28_10_res18_bn_exp.pth)"
echo "cifar10  resnet18    (r34ln->r18)      $(python evaluate.py --data cifar10 --model resnet18    --checkpoint $CKPT/cifar10_res34_ln_res18.pth)"
echo "cifar10  mobilenetv2 (r34ln->mv2)      $(python evaluate.py --data cifar10 --model mobilenetv2 --checkpoint $CKPT/cifar10_res34_ln_mv2.pth)"
echo "cifar100 resnet18    (wrn28->r18)      $(python evaluate.py --data cifar100 --model resnet18   --checkpoint $CKPT/cifar100_wrn_28_10_res18_bn_exp.pth)"
echo "cifar100 wrn_16_8    (wrn28ln->wrn16_8) $(python evaluate.py --data cifar100 --model wrn_16_8  --checkpoint $CKPT/cifar100_wrn28_10_ln_variant_wrn16_8_exp.pth)"
echo "cifar100 resnet18    (wrn28ln->r18)    $(python evaluate.py --data cifar100 --model resnet18   --checkpoint $CKPT/cifar100_wrn28_10_ln_variant_res18_exp.pth)"
echo "imagenet100 resnet18 (r34ln->r18)      $(python evaluate.py --data imagenet100 --model resnet18 --checkpoint $CKPT/imagenet_100_res34_ln_res18_exp.pth --data_path $IMAGENET_VAL --image_size 224)"
