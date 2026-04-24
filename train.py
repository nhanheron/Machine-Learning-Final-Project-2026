"""
Training loop for CheXpert uncertainty-strategy experiments.

Supports all six strategies, including U-MultiClass (CE loss) and
U-SelfTrained (two-pass training with relabelling).
"""

from __future__ import annotations

import copy
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .evaluate import compute_aurocs
from .model import CheXpertModel
from .strategies import (
    apply_strategy,
    apply_strategy_multiclass,
    relabel_uncertain,
)
from .utils import save_checkpoint


# -----------------------------------------------------------------------
# Loss helpers
# -----------------------------------------------------------------------

def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy that ignores masked (NaN / uncertain-ignored) entries."""
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    loss = loss * mask.float()
    if mask.sum() == 0:
        return loss.sum()  # avoid nan from 0/0
    return loss.sum() / mask.sum()


def masked_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int = 3,
) -> torch.Tensor:
    """Per-pathology cross-entropy for U-MultiClass.

    Parameters
    ----------
    logits  : (B, C, 3) – raw logits from the multi-class head
    targets : (B, C)    – LongTensor with class indices 0/1/2
    mask    : (B, C)    – BoolTensor, True where loss should count
    """
    B, C, K = logits.shape
    logits_flat = logits.reshape(B * C, K)
    targets_flat = targets.reshape(B * C)
    mask_flat = mask.reshape(B * C)

    loss = nn.functional.cross_entropy(logits_flat, targets_flat, reduction="none")
    loss = loss * mask_flat.float()
    if mask_flat.sum() == 0:
        return loss.sum()
    return loss.sum() / mask_flat.sum()


# -----------------------------------------------------------------------
# Single-epoch helpers
# -----------------------------------------------------------------------

def _train_one_epoch_binary(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    strategy: str,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0

    for images, raw_labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        raw_labels = raw_labels.to(device, non_blocking=True)

        mapped, mask = apply_strategy(raw_labels, strategy)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = masked_bce_loss(logits, mapped, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def _train_one_epoch_multiclass(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0

    for images, raw_labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        raw_labels = raw_labels.to(device, non_blocking=True)

        targets, mask = apply_strategy_multiclass(raw_labels)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(images)  # (B, C, 3)
            loss = masked_ce_loss(logits, targets, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


# -----------------------------------------------------------------------
# Validation (always binary AUC, even for multiclass – take softmax[:,1])
# -----------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    multiclass: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (mean_auc, all_preds, all_labels)."""
    model.eval()
    all_preds, all_labels = [], []

    for images, raw_labels in tqdm(loader, desc="  val", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)

        if multiclass:
            # Take the "positive" probability from the softmax
            probs = torch.softmax(logits, dim=-1)[:, :, 1]
        else:
            probs = torch.sigmoid(logits)

        all_preds.append(probs.cpu().numpy())
        all_labels.append(raw_labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    aurocs = compute_aurocs(all_labels, all_preds)
    mean_auc = np.nanmean(list(aurocs.values()))
    return mean_auc, all_preds, all_labels


# -----------------------------------------------------------------------
# Main training driver
# -----------------------------------------------------------------------

def train_strategy(
    strategy: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    target_labels: list[str],
    checkpoint_dir: Optional[str] = None,
) -> dict:
    """Train a single strategy end-to-end and return results.

    For ``u_self_trained``, this function handles both passes automatically.

    Returns
    -------
    dict with keys: strategy, best_auc, aurocs, preds, labels
    """
    is_mc = strategy == "u_multiclass"
    use_amp = config["training"]["mixed_precision"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    wd = config["training"]["weight_decay"]

    model = CheXpertModel(
        num_classes=config["model"]["num_classes"],
        multiclass=is_mc,
        pretrained=config["model"]["pretrained"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=use_amp)

    best_auc = 0.0
    best_state = None

    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}  |  Epochs: {epochs}  |  MultiClass: {is_mc}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        if is_mc:
            train_loss = _train_one_epoch_multiclass(
                model, train_loader, optimizer, scaler, device, use_amp
            )
        else:
            train_loss = _train_one_epoch_binary(
                model, train_loader, optimizer, scaler, strategy, device, use_amp
            )

        scheduler.step()
        val_auc, preds, labels = validate(model, val_loader, device, multiclass=is_mc)

        print(
            f"  Epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_auc={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_preds = preds
            best_labels = labels

    # Save checkpoint
    if checkpoint_dir:
        ckpt_path = os.path.join(checkpoint_dir, f"{strategy}_best.pt")
        save_checkpoint(
            {"strategy": strategy, "state_dict": best_state, "auc": best_auc},
            ckpt_path,
        )
        print(f"  Saved checkpoint -> {ckpt_path}")

    aurocs = compute_aurocs(best_labels, best_preds, target_labels)
    print(f"  Best val AUC: {best_auc:.4f}  |  Per-class: {aurocs}")

    return {
        "strategy": strategy,
        "best_auc": best_auc,
        "aurocs": aurocs,
        "preds": best_preds,
        "labels": best_labels,
        "best_state": best_state,
    }


def train_self_trained(
    train_dataset,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    target_labels: list[str],
    checkpoint_dir: Optional[str] = None,
) -> dict:
    """Two-pass U-SelfTrained procedure.

    Pass 1: Train with u_ignore.
    Pass 2: Use the pass-1 model to relabel uncertain entries, then retrain.
    """
    print("\n>>> U-SelfTrained Pass 1 (u_ignore teacher) <<<")
    loader_p1 = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    result_p1 = train_strategy(
        "u_self_trained",  # falls back to u_ignore mapping
        loader_p1,
        val_loader,
        config,
        device,
        target_labels,
    )

    # --- relabel uncertain with teacher predictions ---
    print("\n>>> U-SelfTrained relabelling uncertain samples <<<")
    teacher = CheXpertModel(
        num_classes=config["model"]["num_classes"], pretrained=False
    ).to(device)

    # Load the pass-1 best weights directly from the returned in-memory state
    if result_p1.get("best_state") is not None:
        teacher.load_state_dict(result_p1["best_state"])
    else:
        raise RuntimeError("Pass-1 training did not return a best_state.")

    relabel_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    teacher.eval()
    all_preds = []
    with torch.no_grad():
        for images, _ in tqdm(relabel_loader, desc="  relabel inference"):
            images = images.to(device, non_blocking=True)
            probs = torch.sigmoid(teacher(images))
            all_preds.append(probs.cpu().numpy())
    all_preds = np.concatenate(all_preds)

    new_df = relabel_uncertain(train_dataset.df, all_preds, target_labels)
    new_labels = new_df[target_labels].values
    train_dataset.replace_labels(new_labels)

    # --- Pass 2 with relabelled data ---
    print("\n>>> U-SelfTrained Pass 2 (retrain on relabelled data) <<<")
    loader_p2 = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    # Use u_ones for pass 2 since uncertain labels are now soft probabilities
    result_p2 = train_strategy(
        "u_ones",
        loader_p2,
        val_loader,
        config,
        device,
        target_labels,
        checkpoint_dir,
    )
    result_p2["strategy"] = "u_self_trained"
    return result_p2
