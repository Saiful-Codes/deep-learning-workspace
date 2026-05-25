import collections
import csv
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# Task 1b - LesionDataset
#   Lazy-loading image dataset for the skin-lesion data.
#   The label CSV has 8 columns: image id + 7 one-hot class columns.
#   Only the file names are read in __init__; each image is opened on demand
#   in __getitem__ so the whole dataset never has to fit in memory.
#
# Task 1e - `augment` flag. When True (training set only) a non-deterministic
#   augmentation pipeline is applied; when False (val/test) only a plain
#   Resize + ToTensor is used.


class LesionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, labels_fname, augment=False):
        df = pd.read_csv(labels_fname)
        self.img_dir = Path(img_dir)
        # Store only the file ids now; open the actual images lazily later.
        self.image_ids = df['image'].tolist()
        # One-hot (cols 1..7) -> integer class index, matching IMG_CLASS_NAMES order.
        self.labels = df.iloc[:, 1:].values.argmax(axis=1).astype('int64')

        if augment:
            # Training augmentation: random crop + flips + rotation + colour jitter.
            self.transform = transforms.Compose([
                transforms.Resize((232, 232)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.1, hue=0.02),
                transforms.ToTensor(),
            ])
        else:
            # Deterministic pipeline for validation / test.
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = self.img_dir / f"{self.image_ids[idx]}.jpg"
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), int(self.labels[idx])
