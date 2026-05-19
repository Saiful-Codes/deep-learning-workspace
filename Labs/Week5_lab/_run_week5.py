"""
Week 5 — runnable companion script.
Mirrors the key cells from lab03_solutions.ipynb so we can verify each model
actually trains. Uses local torchvision MNIST instead of the broken URL download
in the original notebook (the CSV mirror at ashwhall.github.io is unreliable).
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -------- Device --------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[setup] device = {device}, torch = {torch.__version__}")

# -------- Data --------
DATA_DIR = os.path.join(os.path.dirname(__file__), "_mnist_data")
os.makedirs(DATA_DIR, exist_ok=True)

tfm = transforms.Compose([transforms.ToTensor()])      # (H,W,1) -> (1,H,W), /255
train_full = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tfm)
test_set   = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)

# Split off a small validation set
g = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(train_full, [55000, 5000], generator=g)

BATCH = 128
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=BATCH, shuffle=False, num_workers=0)

# Peek at a batch
imgs, lbls = next(iter(train_loader))
print(f"[data]  batch images: {imgs.shape} dtype={imgs.dtype} (B,C,H,W)")
print(f"[data]  batch labels: {lbls.shape} dtype={lbls.dtype}")
print(f"[data]  pixel range : [{imgs.min():.3f}, {imgs.max():.3f}]")

# -------- Models --------
class MLP(nn.Module):
    def __init__(self, device, input_size=1*28*28, output_size=10):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )
        self.to(device)
    def forward(self, x): return self.seq(x)


class ConvNet(nn.Module):
    """1 conv block: 28 -> 26 (conv k=3 no pad) -> 13 (pool/2)
       flatten = 8 * 13 * 13 = 1352"""
    def __init__(self, device):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.mlp = MLP(device, input_size=8*13*13)
        self.to(device)
    def forward(self, x): return self.mlp(self.feature_extractor(x))


class DeepConvNet(nn.Module):
    """3 conv blocks:
       28 -> 26 -> 13 -> 11 -> 5 -> 3 -> 1
       flatten = 32 * 1 * 1 = 32"""
    def __init__(self, device):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, 3),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.mlp = MLP(device, input_size=32)
        self.to(device)
    def forward(self, x): return self.mlp(self.feature_extractor(x))


class DropoutMLP(MLP):
    def __init__(self, device, input_size=1*28*28, output_size=10):
        super().__init__(device, input_size, output_size)
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )
        self.to(device)


class DropoutConvNet(DeepConvNet):
    def __init__(self, device):
        super().__init__(device)
        self.mlp = DropoutMLP(device, 32)
        self.to(device)


class BNConvNet(DeepConvNet):
    def __init__(self, device):
        super().__init__(device)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, 3),  nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(32),
        )
        self.to(device)


class SkipBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(out_ch))
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(out_ch))
    def forward(self, x):
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        return x2 + x3            # the skip connection


class SkipConvNet(nn.Module):
    """SkipBlocks use padding=1 -> spatial size preserved.
       Each MaxPool halves: 28 -> 14 -> 7 -> 3 (integer div) -> flatten 32*3*3=288"""
    def __init__(self, device):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            SkipBlock(1, 8),  nn.MaxPool2d(2),
            SkipBlock(8, 16), nn.MaxPool2d(2),
            SkipBlock(16, 32), nn.MaxPool2d(2),
        )
        self.mlp = MLP(device, input_size=32*3*3)
        self.to(device)
    def forward(self, x): return self.mlp(self.feature_extractor(x))


# -------- Training helper --------
def train_one(model_name, model, epochs=1, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(device)
    val_acc   = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(device)

    print(f"\n=== {model_name} ===")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epochs):
        model.train()
        train_acc.reset()
        running = 0.0
        t0 = time.time()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            opt.step()
            running += loss.item()
            train_acc(out, lbls)
        dt = time.time() - t0

        # validation
        model.eval()
        val_acc.reset()
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                val_acc(model(imgs), lbls)

        print(f"  epoch {epoch+1}: train_loss={running/len(train_loader):.4f} "
              f"train_acc={train_acc.compute():.4f} val_acc={val_acc.compute():.4f} "
              f"({dt:.1f}s)")
    return val_acc.compute().item()


# -------- Shape inspection (super-useful for assignments) --------
print("\n[shape] verifying every model passes a (2,1,28,28) tensor cleanly")
dummy = torch.ones(2, 1, 28, 28).to(device)
for name, cls in [
    ("MLP",            MLP),
    ("ConvNet",        ConvNet),
    ("DeepConvNet",    DeepConvNet),
    ("DropoutConvNet", DropoutConvNet),
    ("BNConvNet",      BNConvNet),
    ("SkipConvNet",    SkipConvNet),
]:
    m = cls(device)
    out = m(dummy)
    print(f"  {name:15s}-> {tuple(out.shape)}")


# -------- Run all models for 1 epoch each (~ minutes on GPU) --------
results = {}
for name, cls in [
    ("MLP",            MLP),
    ("ConvNet",        ConvNet),
    ("DeepConvNet",    DeepConvNet),
    ("DropoutConvNet", DropoutConvNet),
    ("BNConvNet",      BNConvNet),
    ("SkipConvNet",    SkipConvNet),
]:
    results[name] = train_one(name, cls(device), epochs=1)

print("\n=== Final val accuracies (1 epoch each) ===")
for name, acc in results.items():
    print(f"  {name:15s}{acc:.4f}")
