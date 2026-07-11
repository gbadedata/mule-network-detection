"""Honest evaluation for mule scoring.

Accuracy is meaningless when 2% of transfers are laundering, so we report precision and
recall of the alerts an investigator would actually work under a fixed budget, the
recall broken out per typology (because the model is strong on some and weak on others),
and the share of laundered value recovered, since a review team cares about money moved,
not transaction counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, scores))


def at_budget(df: pd.DataFrame, scores: np.ndarray, budget: int,
              label_col: str = "is_laundering", amount_col: str = "amount") -> dict:
    """Precision, recall and value recovered if the top `budget` alerts are reviewed."""
    order = np.argsort(-scores)
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[order[:budget]] = True
    y = df[label_col].to_numpy(int)
    amt = df[amount_col].to_numpy(float)

    tp = int((flagged & (y == 1)).sum())
    total_pos = int((y == 1).sum())
    laundered_value = amt[y == 1].sum()
    recovered_value = amt[flagged & (y == 1)].sum()
    return {
        "budget": budget,
        "precision": tp / max(flagged.sum(), 1),
        "recall": tp / max(total_pos, 1),
        "value_recall": float(recovered_value / laundered_value) if laundered_value else 0.0,
    }


def per_typology_recall(df: pd.DataFrame, scores: np.ndarray, budget: int,
                        label_col: str = "is_laundering",
                        pattern_col: str = "pattern") -> pd.DataFrame:
    """Recall of laundering transfers within the top `budget` alerts, per typology."""
    if pattern_col not in df.columns:
        return pd.DataFrame()
    order = np.argsort(-scores)
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[order[:budget]] = True
    ld = df[df[label_col] == 1]
    fl = flagged[df[label_col].to_numpy() == 1]
    out = (pd.DataFrame({"pattern": ld[pattern_col].to_numpy(), "flagged": fl})
           .groupby("pattern")["flagged"].agg(["sum", "count"]))
    out["recall"] = out["sum"] / out["count"]
    return out.reset_index()
