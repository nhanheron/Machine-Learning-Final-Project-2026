"""
Uncertainty-label handling strategies for CheXpert.

Label encoding in the CheXpert CSV:
    1.0  -> positive
    0.0  -> negative
   -1.0  -> uncertain
    NaN  -> unmentioned (always masked out of the loss regardless of strategy)

Each strategy is a callable that takes a raw label tensor and returns:
    (mapped_labels, mask)
where *mask* is a boolean tensor indicating which entries should contribute
to the loss (True = use, False = ignore).
"""

from __future__ import annotations

import torch
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public strategy registry
# ---------------------------------------------------------------------------

STRATEGY_NAMES = [
    "u_ignore",
    "u_zeroes",
    "u_ones",
    "u_multiclass",
    "label_smoothing",
    # u_self_trained is handled at the training level, not here
]


def apply_strategy(
    labels: torch.Tensor,
    strategy: str,
    smooth_pos: float = 0.55,
    smooth_neg: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw CheXpert labels according to the chosen strategy.

    Parameters
    ----------
    labels : Tensor of shape (N, C)
        Raw labels where 1=positive, 0=negative, -1=uncertain, nan=unmentioned.
    strategy : str
        One of the STRATEGY_NAMES (plus "u_self_trained" which falls back to
        u_ignore for the initial training pass).
    smooth_pos / smooth_neg : float
        Soft targets used by the label_smoothing strategy.

    Returns
    -------
    mapped : Tensor (N, C)  – labels ready for BCE loss
    mask   : Tensor (N, C)  – boolean, True where loss should be computed
    """
    nan_mask = torch.isnan(labels)
    uncertain_mask = labels == -1.0

    if strategy in ("u_ignore", "u_self_trained"):
        mask = ~nan_mask & ~uncertain_mask
        mapped = labels.clone()
        mapped[nan_mask | uncertain_mask] = 0.0
        return mapped, mask

    if strategy == "u_zeroes":
        mask = ~nan_mask
        mapped = labels.clone()
        mapped[uncertain_mask] = 0.0
        mapped[nan_mask] = 0.0
        return mapped, mask

    if strategy == "u_ones":
        mask = ~nan_mask
        mapped = labels.clone()
        mapped[uncertain_mask] = 1.0
        mapped[nan_mask] = 0.0
        return mapped, mask

    if strategy == "label_smoothing":
        mask = ~nan_mask
        mapped = labels.clone()
        mapped[uncertain_mask] = smooth_pos
        pos_mask = mapped == 1.0
        neg_mask = mapped == 0.0
        mapped[pos_mask] = 1.0 - smooth_neg
        mapped[neg_mask] = smooth_neg
        mapped[nan_mask] = 0.0
        return mapped, mask

    if strategy == "u_multiclass":
        # Handled separately — returns 3-class targets per pathology.
        # See apply_strategy_multiclass below.
        raise ValueError(
            "u_multiclass must be handled via apply_strategy_multiclass()"
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def apply_strategy_multiclass(
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw labels to 3-class targets for U-MultiClass.

    Returns
    -------
    targets : LongTensor (N, C) with values in {0, 1, 2}
        0 = negative, 1 = positive, 2 = uncertain
    mask : BoolTensor (N, C) – False where label was NaN (unmentioned)
    """
    nan_mask = torch.isnan(labels)
    targets = torch.zeros_like(labels, dtype=torch.long)
    targets[labels == 1.0] = 1
    targets[labels == -1.0] = 2
    targets[nan_mask] = 0
    mask = ~nan_mask
    return targets, mask


# ---------------------------------------------------------------------------
# Helpers for U-SelfTrained relabelling
# ---------------------------------------------------------------------------

def relabel_uncertain(
    df: pd.DataFrame,
    predictions: np.ndarray,
    target_labels: list[str],
) -> pd.DataFrame:
    """Replace uncertain (-1) entries in *df* with model predictions.

    Parameters
    ----------
    df : DataFrame with the original CheXpert CSV columns.
    predictions : ndarray (N, C) of sigmoid probabilities from the teacher model.
    target_labels : list of column names corresponding to the C pathologies.

    Returns
    -------
    A copy of df with -1 entries replaced by the predicted probabilities.
    """
    df = df.copy()
    for i, col in enumerate(target_labels):
        uncertain = df[col] == -1.0
        df.loc[uncertain, col] = predictions[uncertain.values, i]
    return df
