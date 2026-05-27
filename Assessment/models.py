import torch
import torch.nn as nn
import torchvision.models as tv_models


class SimpleBNConv(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(kernel_size=2)
            )

        self.features = nn.Sequential(
            conv_block(3, 8),
            conv_block(8, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_resnet18(num_classes=7, freeze=True):
    """
    Build a pretrained ResNet18 model for 7-class lesion classification.

    If freeze=True:
        freeze all pretrained feature layers and train only the final layer.

    If freeze=False:
        fine-tune all layers.
    """

    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# TODO Task 1g - Create your own models



