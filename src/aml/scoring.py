"""Account-level prioritisation for investigators.

Transaction scores are turned into account risk (an account is as risky as its riskiest
transfer, sent or received), then ranked by expected laundered value, risk times the
money the account moved, so a review team with a fixed budget works the accounts where
the most dirty money is likely to be. Each row carries plain-language reason codes drawn
from the structural signals, so an alert explains itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graph_features import structural_flags


def account_risk(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Per-account risk: the maximum score of any transfer it sent or received."""
    s = pd.DataFrame({
        "from_account": df["from_account"].to_numpy(),
        "to_account": df["to_account"].to_numpy(),
        "score": scores,
    })
    as_src = s.groupby("from_account")["score"].max()
    as_dst = s.groupby("to_account")["score"].max()
    accts = as_src.index.union(as_dst.index)
    risk = np.maximum(as_src.reindex(accts).fillna(0.0),
                      as_dst.reindex(accts).fillna(0.0))
    return pd.DataFrame({"account": accts, "risk": risk.to_numpy()})


def _reasons(row) -> str:
    out = []
    if row.fan_in_flag:
        out.append(f"fans in: {int(row.max_in_burst)} inflows in a short window")
    if row.fan_out_flag:
        out.append(f"fans out: {int(row.max_out_burst)} outflows in a short window")
    if row.passthrough_flag:
        out.append(f"u-turn: forwards funds it just received (match {row.rapid_passthrough:.2f})")
    if not out:
        out.append("elevated model score without a single clear structural pattern")
    return "; ".join(out)


def account_queue(df: pd.DataFrame, scores: np.ndarray, top: int = 50) -> pd.DataFrame:
    """Ranked investigation queue: accounts by expected laundered value, with reasons."""
    s = structural_flags(df)
    q = s.merge(account_risk(df, scores), on="account", how="left")
    q["risk"] = q["risk"].fillna(0.0)
    q["value_moved"] = np.maximum(q["amt_in"], q["amt_out"])
    # burst concentration: peak burst over total activity. Near 1 for a mule whose
    # activity is packed into one window, near 0 for a legit account busy all the time.
    activity = (q["in_cnt"] + q["out_cnt"]).clip(lower=1)
    q["concentration"] = np.maximum(q["max_in_burst"], q["max_out_burst"]) / activity
    # log-dollars, not raw dollars, so a handful of high-throughput accounts do not
    # dominate purely on size; risk and concentration then carry weight.
    q["priority"] = q["risk"] * np.log1p(q["value_moved"]) * q["concentration"]
    q = q.sort_values("priority", ascending=False).head(top).reset_index(drop=True)
    q["reasons"] = [_reasons(r) for r in q.itertuples(index=False)]
    cols = ["account", "priority", "risk", "value_moved", "concentration",
            "max_in_burst", "max_out_burst", "is_laundering_acct", "reasons"]
    return q[cols]
