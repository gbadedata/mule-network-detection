"""Supervised scoring over the leakage-safe streaming features.

A time split (train on the earliest transfers, test on the latest) so we never train
on the future, matching how this would run in production. The model is deliberately
feature-agnostic: it takes a `feature_cols` list, so the same code scores any feature
set. Laundering is rare, so we weight the positive class rather than resampling.

The point is not a single number. It is that combining the graph-derived signals in one
model beats every single hand threshold, and that the way to read it is per typology and
by value recovered under an investigation budget, not by accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

DEFAULTS = dict(
    max_leaf_nodes=31,
    learning_rate=0.1,
    max_iter=250,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=0,
)


def time_split(df: pd.DataFrame, train_frac: float = 0.6, valid_frac: float = 0.2):
    """Split by time into train / valid / test with no shuffling."""
    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)
    a, b = int(n * train_frac), int(n * (train_frac + valid_frac))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def train_model(train_df: pd.DataFrame, feature_cols: list[str],
                label_col: str = "is_laundering", **params) -> HistGradientBoostingClassifier:
    """Fit a gradient-boosted classifier on the given features, weighting positives."""
    cfg = {**DEFAULTS, **params}
    x = train_df[feature_cols].to_numpy(np.float32)
    y = train_df[label_col].to_numpy(int)
    pos = max(int(y.sum()), 1)
    neg = max(len(y) - pos, 1)
    w = np.where(y == 1, neg / pos, 1.0)
    model = HistGradientBoostingClassifier(**cfg)
    model.fit(x, y, sample_weight=w)
    return model


def score(model: HistGradientBoostingClassifier, df: pd.DataFrame,
          feature_cols: list[str]) -> np.ndarray:
    """Return the laundering probability for each row."""
    return model.predict_proba(df[feature_cols].to_numpy(np.float32))[:, 1]
