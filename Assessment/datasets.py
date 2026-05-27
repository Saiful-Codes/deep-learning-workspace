import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms

from pathlib import Path
from PIL import Image


class LesionDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        img_dir,
        labels_fname,
        augment=False,
        augmentation_type="none",
        pretrained=False
    ):

        self.img_dir = Path(img_dir)

        df = pd.read_csv(labels_fname)

        self.image_ids = df["image"].tolist()

        self.labels = (
            df.iloc[:, 1:]
            .values
            .argmax(axis=1)
            .astype("int64")
        )

        # ImageNet normalization for pretrained models
        imagenet_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Base transforms
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]

        if pretrained:
            base_transforms.append(imagenet_norm)

        # No augmentation
        if not augment or augmentation_type == "none":

            self.transform = transforms.Compose(base_transforms)

        # Horizontal flip only
        elif augmentation_type == "hflip":

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                imagenet_norm if pretrained else transforms.Identity()
            ])

        # Rotation only
        elif augmentation_type == "rotation":

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=20),
                transforms.ToTensor(),
                imagenet_norm if pretrained else transforms.Identity()
            ])

        # ColorJitter only
        elif augmentation_type == "colorjitter":

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                ),
                transforms.ToTensor(),
                imagenet_norm if pretrained else transforms.Identity()
            ])

        # Full augmentation pipeline
        elif augmentation_type == "full":

            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                ),
                transforms.RandomResizedCrop(
                    size=224,
                    scale=(0.8, 1.0)
                ),
                transforms.ToTensor(),
                imagenet_norm if pretrained else transforms.Identity()
            ])

        else:
            raise ValueError(
                f"Unknown augmentation_type: {augmentation_type}"
            )

    def __len__(self):

        return len(self.image_ids)

    def __getitem__(self, idx):

        img_path = self.img_dir / f"{self.image_ids[idx]}.jpg"

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        label = int(self.labels[idx])

        return image, label