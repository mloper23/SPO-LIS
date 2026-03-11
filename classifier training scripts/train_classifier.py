from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import WeightedRandomSampler
import plotly.graph_objects as go

import json
from datetime import datetime

# Change this to your dataset folder (the one that contains metadata.csv, kernel.png, patches_npz/)
DATASET_DIR = Path('./dataset')  # <-- change if needed

META_CSV = DATASET_DIR / "metadata.csv"
PATCHES_DIR = DATASET_DIR / "patches_npz"
# df = pd.read_csv(META_CSV, engine="python", on_bad_lines="skip")
# df.shape
df = pd.read_csv(META_CSV)
df.head(), df.shape


# Ensure patch_file is an absolute path
# In your latest script you stored patch_file as str(patch_file) (absolute). If it's only a name, fix it:
def resolve_patch_path(p):
    p = str(p)
    pp = Path(p)
    if pp.exists():
        return pp
    # If stored as filename only, resolve relative to patches dir
    return (PATCHES_DIR / pp.name)

df["patch_path"] = df["patch_file"].apply(resolve_patch_path)

# Drop rows with missing files (just in case)
exists_mask = df["patch_path"].apply(lambda p: Path(p).exists())
df = df[exists_mask].reset_index(drop=True)

# Make label -> index mapping
labels = sorted(df["label"].unique())
label2idx = {lab: i for i, lab in enumerate(labels)}
idx2label = {i: lab for lab, i in label2idx.items()}

df["y"] = df["label"].map(label2idx).astype(int)

print("Classes:", labels)
print("Rows:", len(df))
df[["label","y","patch_index","patch_path"]].head()


# Stratified split keeps class balance
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["y"]
)
train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df["y"]
)

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
print("Train class counts:\n", train_df["label_unified"].value_counts())

class NPZPatchDataset(Dataset):
    def __init__(self, df, num_channels=None, normalize=True, cache_size=4):
        """
        df must contain: patch_path, patch_index, y
        """
        self.df = df.reset_index(drop=True)
        self.normalize = normalize
        self.num_channels = num_channels
        self.cache_size = cache_size

        self._cache = {}         # path -> patches array
        self._cache_order = []   # LRU order

    def _get_npz_patches(self, path: Path):
        path = Path(path)

        if path in self._cache:
            # refresh LRU
            self._cache_order.remove(path)
            self._cache_order.append(path)
            return self._cache[path]

        data = np.load(path, allow_pickle=True)
        patches = data["patches"]   # expected shape (K, C, H, W) or (K, H, W) depending on your save
        data.close()

        # Ensure 4D: (K,C,H,W)
        if patches.ndim == 3:
            patches = patches[:, None, :, :]

        # Optional: keep only first num_channels
        if self.num_channels is not None:
            patches = patches[:, :self.num_channels]

        # Add to cache (LRU)
        self._cache[path] = patches
        self._cache_order.append(path)
        if len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

        return patches

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row["patch_path"])
        pidx = int(row["patch_index"])
        y = int(row["y"])

        patches = self._get_npz_patches(path)
        x = patches[pidx]  # (C,H,W)

        # Convert to float tensor
        x = torch.from_numpy(x).float()

        # If uint8 images 0..255, normalize to 0..1
        if self.normalize:
            # if values look like uint8
            if x.max() > 1.5:
                x = x / 255.0

        return x, y


train_ds = NPZPatchDataset(train_df, normalize=True, cache_size=8)
val_ds   = NPZPatchDataset(val_df, normalize=True, cache_size=8)
test_ds  = NPZPatchDataset(test_df, normalize=True, cache_size=8)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

# Check one batch
x, y = next(iter(train_loader))
print("Batch x:", x.shape, x.dtype)  # (B,C,H,W)
print("Batch y:", y.shape, y.dtype)

# train_df must have integer labels in column "y"
class_counts = train_df["y"].value_counts().sort_index().values
n_classes = len(class_counts)

# weight per class = inverse frequency
class_weights = 1.0 / class_counts
print("class_counts:", class_counts)
print("class_weights:", class_weights)

# weight per sample
sample_weights = class_weights[train_df["y"].values]
sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_df),   # draws this many samples per epoch
    replacement=True
)


class SimpleCNN(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # /8

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)

# infer channels/classes
in_ch = x.shape[1]
n_classes = len(labels)

model = SimpleCNN(in_ch=in_ch, n_classes=n_classes)
model

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.8e-3)

def run_epoch(loader, train=True):
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)

    return total_loss / total, correct / total

# ---- train + log ----
history = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
}

best_val = 0.0
best_state = None

epochs = 150
for ep in range(1, epochs + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader, train=False)

    history["epoch"].append(ep)
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)

    if va_acc > best_val:
        best_val = va_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

# load best model
if best_state is not None:
    model.load_state_dict(best_state)


fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=history["epoch"], y=history["train_loss"],
                              mode="lines+markers", name="Train Loss"))
fig_loss.add_trace(go.Scatter(x=history["epoch"], y=history["val_loss"],
                              mode="lines+markers", name="Val Loss"))
fig_loss.update_layout(title="Loss vs Epoch", xaxis_title="Epoch", yaxis_title="Loss",
                       width=850, height=450)
# fig_loss.show()

# ---- Plotly: Accuracy ----
fig_acc = go.Figure()
fig_acc.add_trace(go.Scatter(x=history["epoch"], y=history["train_acc"],
                             mode="lines+markers", name="Train Accuracy"))
fig_acc.add_trace(go.Scatter(x=history["epoch"], y=history["val_acc"],
                             mode="lines+markers", name="Val Accuracy"))
fig_acc.update_layout(title="Accuracy vs Epoch", xaxis_title="Epoch", yaxis_title="Accuracy",
                      width=850, height=450, yaxis=dict(range=[0, 1]))
# fig_acc.show()
