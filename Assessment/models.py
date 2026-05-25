import torch
import torch.nn as nn
import torchvision

# Task 1c - Baseline convolutional network.
#   5 conv blocks with output channels 8, 16, 32, 64, 128.
#   Each block: Conv2d(3x3, pad=1) -> ReLU -> BatchNorm2d -> MaxPool2d(2).
#   With a 224x224 input each block halves the spatial size:
#       224 -> 112 -> 56 -> 28 -> 14 -> 7, leaving a 128 x 7 x 7 feature map,
#   which is flattened and mapped to num_classes by a single Linear layer.
class SimpleBNConv(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_ch),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 8),      # 224 -> 112
            block(8, 16),     # 112 -> 56
            block(16, 32),    #  56 -> 28
            block(32, 64),    #  28 -> 14
            block(64, 128),   #  14 -> 7
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


# TODO Task 1f - Create a model from a pre-trained model from the torchvision
#  model zoo.


# TODO Task 1g - Create your own models

