"""
CheXpert PyTorch Dataset.

Reads the CheXpert CSV, loads images, and returns (image, raw_labels) pairs.
Label strategy mapping is applied at training time so the same dataset object
can be reused across strategies.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ImageNet normalisation stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TARGET_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CheXpertDataset(Dataset):
    """
    Parameters
    ----------
    csv_path : str
        Path to train.csv or valid.csv.
    data_dir : str
        Root directory that the image paths in the CSV are relative to.
        For the small dataset the CSV paths look like
        ``CheXpert-v1.0-small/train/patient.../study.../view.jpg``.
        Set *data_dir* to the **parent** of ``CheXpert-v1.0-small/``.
    target_labels : list[str]
        Which pathology columns to include.
    transform : torchvision transform, optional
    """

    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        target_labels: Optional[list[str]] = None,
        transform: Optional[transforms.Compose] = None,
        subset_frac: Optional[float] = None,
        subset_seed: int = 42,
    ):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.target_labels = target_labels or TARGET_LABELS
        self.transform = transform

        # Keep only frontal views for simplicity (AP / PA).
        if "Frontal/Lateral" in self.df.columns:
            self.df = self.df[self.df["Frontal/Lateral"] == "Frontal"].reset_index(
                drop=True
            )

        # Optional random subset — useful for fast iteration on Colab.
        if subset_frac is not None and 0 < subset_frac < 1.0:
            self.df = self.df.sample(
                frac=subset_frac, random_state=subset_seed
            ).reset_index(drop=True)

        self.labels = self.df[self.target_labels].values.astype(np.float32)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["Path"])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_vec = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label_vec

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def replace_labels(self, new_labels: np.ndarray):
        """Hot-swap labels (used by U-SelfTrained after relabelling)."""
        assert new_labels.shape == self.labels.shape
        self.labels = new_labels.astype(np.float32)
        # Also update the dataframe so downstream code stays consistent
        for i, col in enumerate(self.target_labels):
            self.df[col] = new_labels[:, i]
