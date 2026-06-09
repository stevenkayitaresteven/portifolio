"""Model factory for the explicit-content classifier.

We fine-tune a MobileNetV3-Large backbone (a good speed/accuracy trade-off for
CPU inference) pretrained on ImageNet, replacing the classifier head with a
multi-label head sized to the dataset's classes. MobileNetV3 keeps the exported
ONNX small (~12 MB) and fast on CPU, matching this project's constraints.

``efficientnet_b0`` and ``resnet18`` are offered as alternatives.
"""
from __future__ import annotations


def build_model(num_classes: int, arch: str = "mobilen_v3_large",
                pretrained: bool = True):
    import torch.nn as nn
    from torchvision import models

    arch = arch.lower()
    if arch in ("mobilen_v3_large", "mobilenetv3", "mobilenet_v3_large"):
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch in ("efficientnet_b0", "efficientnet"):
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch in ("resnet18", "resnet"):
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown arch: {arch}")
    return net


def freeze_backbone(net, arch: str = "mobilen_v3_large"):
    """Freeze feature-extractor weights so only the head trains first.

    Useful for tiny datasets: train the head for a couple of epochs, then call
    :func:`unfreeze` to fine-tune the whole network at a lower LR.
    """
    for p in net.parameters():
        p.requires_grad = False
    # Re-enable just the classification head.
    head = getattr(net, "fc", None) or getattr(net, "classifier", None)
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True
    return net


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad = True
    return net
